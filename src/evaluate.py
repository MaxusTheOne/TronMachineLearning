#!/usr/bin/env python3
"""
Evaluate different Tron policies head‑to‑head in 1v1 matches.
Measures win rates and average survival steps using parallel environments.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Callable, Any

from model import TronBatchModel
from controller import RandomController, GreedySpaceController
from agent_classes import MLPPolicy
from stable_baselines3 import PPO

# -------------------------------------------------------------------
# Wrappers to unify the .actions(model) interface for single‑player control
# -------------------------------------------------------------------

class RLPolicyWrapper:
    """Adapts a PPO model to control a specific player (default player 0)."""
    def __init__(self, model_path: str, player_id: int = 0, device: str = "auto"):
        self.model = PPO.load(model_path, device=device)
        self.player_id = player_id

    def actions(self, model: TronBatchModel) -> np.ndarray:
        """
        Returns actions for ALL players in the batch, shape (envs, players).
        Only the actions for self.player_id come from the PPO; other players get 0.
        """
        envs, players = model.envs, model.players
        obs = model.observe_lite()                     # (envs, players, obs_dim)
        acts = np.zeros((envs, players), dtype=np.int8)
        for e in range(envs):
            action, _ = self.model.predict(obs[e, self.player_id], deterministic=True)
            acts[e, self.player_id] = action
        return acts


class GAPolicyWrapper:
    """Adapts a MLPPolicy genome to control a specific player."""
    def __init__(self, genome: np.ndarray, obs_dim: int, hidden: int, player_id: int = 0):
        self.policy = MLPPolicy(obs_dim, hidden=hidden, genome=genome)
        self.player_id = player_id

    def actions(self, model: TronBatchModel) -> np.ndarray:
        """Returns full action matrix (envs, players) but only player_id uses the GA."""
        full_actions = self.policy.actions(model)      # already (envs, players)
        # If the GA policy already controls all players symmetrically, we keep its output.
        # To restrict it to only one player, zero out other columns:
        mask = np.ones_like(full_actions, dtype=bool)
        mask[:, self.player_id] = False
        full_actions[mask] = 0
        return full_actions


class SinglePlayerControllerWrapper:
    """Wraps any controller that normally outputs actions for ALL players,
    but restricts it to control only a single player (others become 0)."""
    def __init__(self, controller: Any, player_id: int = 0):
        self.controller = controller
        self.player_id = player_id

    def actions(self, model: TronBatchModel) -> np.ndarray:
        full = self.controller.actions(model)          # shape (envs, players)
        mask = np.ones_like(full, dtype=bool)
        mask[:, self.player_id] = False
        full[mask] = 0
        return full


# -------------------------------------------------------------------
# Core match function
# -------------------------------------------------------------------
def run_match(
    policy0: Any,
    policy1: Any,
    *,
    envs: int = 2048,
    width: int = 64,
    height: int = 48,
    max_steps: Optional[int] = None,
    seed: Optional[int] = None,
) -> Dict[str, float]:
    """
    Runs many parallel episodes of policy0 (player 0) vs policy1 (player 1).
    Returns win rates and average survival steps for both.
    """
    players = 2
    max_steps = max_steps or (width * height)

    model = TronBatchModel(
        width=width, height=height, players=players, envs=envs,
        max_steps=max_steps, keep_owner=False, seed=seed
    )

    # Stats
    wins = np.zeros((envs, players), dtype=np.float32)
    death_step = np.full((envs, players), -1, dtype=np.int32)   # -1 = not dead yet

    # Episode length for finished envs
    episode_length = np.zeros(envs, dtype=np.int32)

    # Reset all
    model.reset()
    step = 0

    # We'll also track whether each env has finished
    finished_episodes = np.zeros(envs, dtype=bool)

    while not np.all(finished_episodes):
        # Get actions
        actions0 = policy0.actions(model)
        actions1 = policy1.actions(model)
        combined = actions0.copy()
        combined[:, 1] = actions1[:, 1]

        result = model.step(combined)

        # Update step counter
        step += 1

        # Check for newly dead players
        died = result.died & ~finished_episodes[:, None]
        if died.any():
            death_step[died] = step

        # Check for finished episodes (done and not already recorded)
        just_finished = result.done & ~finished_episodes
        if just_finished.any():
            for e in np.where(just_finished)[0]:
                finished_episodes[e] = True
                episode_length[e] = step
                alive = result.alive[e]
                if np.sum(alive) == 1:
                    winner = np.argmax(alive)
                    wins[e, winner] = 1.0

        # Auto‑reset finished envs inside the model (for next step, they will be fresh)
        model.auto_reset_done()

    # Compute survival steps for each player
    # For players that died, use death_step; for survivors (not dead by end), use episode_length
    survival0 = np.where(death_step[:, 0] >= 0, death_step[:, 0], episode_length)
    survival1 = np.where(death_step[:, 1] >= 0, death_step[:, 1], episode_length)

    win_rate0 = wins[:, 0].mean()
    win_rate1 = wins[:, 1].mean()
    draw_rate = 1.0 - (win_rate0 + win_rate1)

    return {
        "win_rate_0": win_rate0,
        "win_rate_1": win_rate1,
        "draw_rate": draw_rate,
        "avg_survival_0": survival0.mean(),
        "avg_survival_1": survival1.mean(),
    }

def compare_all_policies(
    rl_model_path: str,
    ga_genome_path: str,
    ga_hidden: int,
    obs_dim: int,
    num_episodes: int = 1000,
    envs_parallel: int = 1024,
    width: int = 64,
    height: int = 48,
    seed: int = 42,
) -> None:
    """
    Loads RL, GA, Random, Greedy and runs every pair (0 vs 1) matchup.
    Prints a table of win rates and survival turns.
    """
    # Load policies (controlling player 0)
    rl = RLPolicyWrapper(rl_model_path, player_id=0)
    genome = np.load(ga_genome_path)
    ga = GAPolicyWrapper(genome, obs_dim, ga_hidden, player_id=0)
    random = SinglePlayerControllerWrapper(RandomController(seed=seed), player_id=0)
    greedy = SinglePlayerControllerWrapper(GreedySpaceController(), player_id=0)

    policies = {
        "RL": rl,
        "GA": ga,
        "Random": random,
        "Greedy": greedy,
    }

    results = {}
    for name0, pol0 in policies.items():
        for name1, pol1 in policies.items():
            if name0 == name1:
                continue
            # For each ordered pair, run match with pol0 as player0, pol1 as player1
            stats = run_match(
                pol0, pol1,
                envs=envs_parallel,
                width=width, height=height,
                max_steps=width*height,
                seed=seed
            )
            results[(name0, name1)] = stats

    # Print formatted results
    print("\n=== Tournament Results (player0 vs player1) ===\n")
    for (p0, p1), s in results.items():
        print(f"{p0:8} vs {p1:8} : "
              f"Win {p0}: {s['win_rate_0']*100:5.1f}%  "
              f"Win {p1}: {s['win_rate_1']*100:5.1f}%  "
              f"Draw: {s['draw_rate']*100:5.1f}%  |  "
              f"Avg turns: {p0} {s['avg_survival_0']:.1f}  vs  {p1} {s['avg_survival_1']:.1f}")


if __name__ == "__main__":
    # Paths to your saved models (adjust!)
    RL_PATH = "ppos/ppo4.zip"
    GA_GENOME_PATH = "tron_genomes/tron_p2_32x32_obs13_h64_pop64_gen20_eval2048_seed42_20260507_111737.npy"   # from your notebook
    GA_HIDDEN = 64                # must match training
    OBS_DIM = 13                  # for 64x48 board, observe_lite gives 13 features

    compare_all_policies(
        rl_model_path=RL_PATH,
        ga_genome_path=GA_GENOME_PATH,
        ga_hidden=GA_HIDDEN,
        obs_dim=OBS_DIM,
        num_episodes=2000,        # total episodes per matchup (summed over parallel envs)
        envs_parallel=1024,       # run 1024 games in parallel for speed
        width=64, height=48,
        seed=42
    )