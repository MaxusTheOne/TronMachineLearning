#!/usr/bin/env python3
"""
Train a single‑agent reinforcement learning policy for the Tron (Lightcycles) game.

The environment uses the fast vectorized TronBatchModel from the provided
'tron_model.py'. The agent controls player 0, while opponents (player 1 and
optionally more) follow a fixed policy (random or simple heuristic).

Dependencies:
    numpy, gymnasium, stable-baselines3, torch

Usage:
    python train_tron_rl.py --total_timesteps 1_000_000 --model_path tron_ppo.zip
"""

import argparse
from typing import Optional, Tuple, Dict, Any

import gymnasium as gym
from gymnasium import spaces
import numpy as np

from batch_model import TronBatchModel  # assume the provided file is named tron_model.py


class FixedOpponentPolicy:
    """Base class for opponent policies. Must implement act(obs, model)."""

    def act(self, model: TronBatchModel, env_id: int = 0) -> np.ndarray:
        """
        Return actions for all opponents (players 1..P-1) in a given environment.

        Args:
            model: TronBatchModel instance (batch size >= 1). Only env_id is used.
            env_id: index of the environment within the batch.

        Returns:
            action array of shape (P-1,) with values in {0,1,2}
        """
        raise NotImplementedError


class RandomOpponent(FixedOpponentPolicy):
    """Opponents choose uniformly random legal actions, or straight if none legal."""

    def act(self, model: TronBatchModel, env_id: int = 0) -> np.ndarray:
        legal = model.legal_actions()[env_id]           # shape (players, 3)
        n_opponents = model.players - 1
        acts = np.empty(n_opponents, dtype=np.int8)
        for i in range(n_opponents):
            player = i + 1
            legal_actions = np.where(legal[player])[0]
            if len(legal_actions) == 0:
                acts[i] = 0   # fallback – all moves are illegal, game about to end
            else:
                acts[i] = np.random.choice(legal_actions)
        return acts


class GreedyOpponent(FixedOpponentPolicy):
    """
    Opponents try to move towards the center of the board, unless that move is illegal.
    Falls back to random legal move if the preferred direction is not available.
    """

    def act(self, model: TronBatchModel, env_id: int = 0) -> np.ndarray:
        legal = model.legal_actions()[env_id]
        n_opponents = model.players - 1
        acts = np.empty(n_opponents, dtype=np.int8)
        center_x = model.width // 2
        center_y = model.height // 2

        for i in range(n_opponents):
            player = i + 1
            if not model.alive[env_id, player]:
                # dead opponents don't act (their actions are ignored by step anyway)
                acts[i] = 0
                continue

            # Current position and heading
            px, py = model.pos[env_id, player]
            heading = model.heading[env_id, player]

            # Evaluate each action: resulting direction and position
            best_action = 0
            best_distance = float('inf')
            for action in [0, 1, 2]:
                if not legal[player, action]:
                    continue
                new_heading = (heading + [0, -1, 1][action]) & 3
                dx, dy = [(0, -1), (1, 0), (0, 1), (-1, 0)][new_heading]
                nx, ny = px + dx, py + dy
                dist = abs(nx - center_x) + abs(ny - center_y)
                if dist < best_distance:
                    best_distance = dist
                    best_action = action
            if best_distance == float('inf'):
                # no legal move – choose any (will cause death)
                best_action = 0
            acts[i] = best_action
        return acts


class TronSingleAgentEnv(gym.Env):
    """
    Gymnasium environment for a single RL agent controlling player 0 in Tron.

    The environment wraps the vectorized TronBatchModel with envs=1 (single game).
    Opponents (players 1..players-1) act according to a fixed policy.
    """

    metadata = {"render_modes": ["rgb_array", "human"], "render_fps": 30}

    def __init__(
        self,
        width: int = 48,
        height: int = 32,
        players: int = 2,
        max_steps: Optional[int] = None,
        opponent_policy: str = "random",
        render_mode: Optional[str] = None,
        seed: Optional[int] = None,
    ):
        """
        Args:
            width: board width
            height: board height
            players: total number of players (including agent)
            max_steps: maximum steps per episode (default width*height)
            opponent_policy: "random" or "greedy"
            render_mode: "rgb_array" or "human" (requires matplotlib)
            seed: random seed for environment and opponent
        """
        super().__init__()
        self.width = width
        self.height = height
        self.players = players
        self.max_steps = max_steps or (width * height)
        self.render_mode = render_mode
        self.seed = seed

        # Internal vectorized model (single environment)
        self.model = TronBatchModel(
            width=width,
            height=height,
            players=players,
            envs=1,
            max_steps=self.max_steps,
            keep_owner=True,          # needed for rendering
            randomize_spawns=True,
            seed=seed,
        )

        # Opponent policy
        if opponent_policy == "random":
            self.opponent = RandomOpponent()
        elif opponent_policy == "greedy":
            self.opponent = GreedyOpponent()
        else:
            raise ValueError(f"Unknown opponent policy: {opponent_policy}")

        # Observation space: based on `observe_lite` for player 0 only
        # Determine feature dimension by calling the method once
        self.model.reset()
        obs_lite = self.model.observe_lite()[0, 0]   # [env=0, player=0]
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=obs_lite.shape, dtype=np.float32
        )

        # Action space: straight / left / right
        self.action_space = spaces.Discrete(3)

        # For rendering
        self._render_image = None

    def reset(
        self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset the game. Returns observation for player 0."""
        super().reset(seed=seed)
        if seed is not None:
            self.model.rng = np.random.default_rng(seed)

        self.model.reset(env_ids=[0])
        obs = self.model.observe_lite()[0, 0]
        info = {}
        return obs, info

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        Apply an action for the agent, get opponent actions, and advance environment.

        Returns:
            obs: observation for agent (player 0)
            reward: scalar reward for agent
            terminated: whether episode ended (win/loss/draw)
            truncated: never True (max steps handled internally)
            info: additional info (winner, etc.)
        """
        # Build action array for all players
        actions = np.zeros((1, self.players), dtype=np.int8)
        actions[0, 0] = action
        # Opponent actions
        opponents_actions = self.opponent.act(self.model, env_id=0)
        actions[0, 1:] = opponents_actions

        # Execute step
        result = self.model.step(actions)

        # Agent reward: result.reward[0,0] (scalar)
        reward = float(result.reward[0, 0])

        # Termination: game done or agent died
        done_game = result.done[0]
        agent_alive = result.alive[0, 0]
        terminated = done_game or not agent_alive
        truncated = False   # max_steps already handled by model

        info = {
            "done": bool(result.done[0]),
            "alive": result.alive[0].copy(),
            "winner": None,
        }
        if done_game:
            # Determine winner: if exactly one alive, its index; else draw
            alive_players = np.where(result.alive[0])[0]
            if len(alive_players) == 1:
                info["winner"] = int(alive_players[0])
            else:
                info["winner"] = -1  # draw

        # Get new observation for the agent
        obs = self.model.observe_lite()[0, 0]

        return obs, reward, terminated, truncated, info

    def render(self) -> Optional[np.ndarray]:
        """Render the current board. If render_mode is 'rgb_array', return an RGB image."""
        if self.render_mode is None:
            return None

        # Use the owner grid for colourful rendering
        owner = self.model.owner[0] if self.model.owner is not None else None
        if owner is None:
            # Fallback: generate owner from occupied and head positions
            owner = np.zeros((self.height, self.width), dtype=np.uint8)
            # Mark trails by player id (only positions occupied by heads are known without owner)
            # This is crude; better to set keep_owner=True in model.
            for p in range(self.players):
                if self.model.alive[0, p]:
                    px, py = self.model.pos[0, p]
                    if 0 <= px < self.width and 0 <= py < self.height:
                        owner[py, px] = p + 1
        else:
            owner = owner.copy()

        # Colour map for players (BGR for OpenCV style, but matplotlib uses RGB)
        # Player 0: cyan, Player 1: magenta, Player 2: yellow, Player 3: lime
        colours = np.array([
            [0, 0, 0],        # 0 empty
            [0, 255, 255],    # 1 cyan
            [255, 0, 255],    # 2 magenta
            [0, 255, 0],      # 3 lime
            [255, 255, 0],    # 4 yellow (if more players)
        ], dtype=np.uint8)

        img = colours[owner]   # (H, W, 3)
        # Draw heads with brighter colour (or circles) – just use same colour for simplicity
        # For better visibility, we can draw a white dot on head positions
        for p in range(self.players):
            if self.model.alive[0, p]:
                px, py = self.model.pos[0, p]
                if 0 <= px < self.width and 0 <= py < self.height:
                    # Make head pixel white
                    img[py, px] = [255, 255, 255]

        if self.render_mode == "rgb_array":
            return img
        elif self.render_mode == "human":
            try:
                import matplotlib.pyplot as plt
                if not hasattr(self, "_fig"):
                    plt.ion()
                    self._fig, self._ax = plt.subplots(figsize=(self.width/4, self.height/4))
                    self._ax.set_title("Tron Game")
                    self._img_handle = self._ax.imshow(img, interpolation="nearest")
                    plt.show()
                else:
                    self._img_handle.set_data(img)
                    self._fig.canvas.flush_events()
            except ImportError:
                print("matplotlib not installed; cannot render to human window.")
        return None

    def close(self):
        if hasattr(self, "_fig"):
            plt.close(self._fig)


def main():
    parser = argparse.ArgumentParser(description="Train a PPO agent for Tron.")
    parser.add_argument("--total_timesteps", type=int, default=1_000_000,
                        help="Number of environment steps to train for.")
    parser.add_argument("--model_path", type=str, default="tron_ppo.zip",
                        help="Path to save the trained model.")
    parser.add_argument("--width", type=int, default=48, help="Board width.")
    parser.add_argument("--height", type=int, default=32, help="Board height.")
    parser.add_argument("--players", type=int, default=2, help="Total players (including agent).")
    parser.add_argument("--opponent", type=str, default="random", choices=["random", "greedy"],
                        help="Opponent policy.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--learning_rate", type=float, default=3e-4, help="PPO learning rate.")
    parser.add_argument("--n_steps", type=int, default=2048, help="PPO steps per update.")
    parser.add_argument("--batch_size", type=int, default=64, help="PPO batch size.")
    parser.add_argument("--n_epochs", type=int, default=10, help="PPO epochs per update.")
    parser.add_argument("--render", action="store_true", help="Render during evaluation after training.")
    args = parser.parse_args()

    # Create the environment
    env = TronSingleAgentEnv(
        width=args.width,
        height=args.height,
        players=args.players,
        opponent_policy=args.opponent,
        render_mode="human" if args.render else None,
        seed=args.seed,
    )

    # Wrap for SB3: use DummyVecEnv to enable batch training
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv
    from stable_baselines3.common.monitor import Monitor

    # Monitor for logging episode statistics
    env = Monitor(env)
    vec_env = DummyVecEnv([lambda: env])

    # Configure PPO
    policy_kwargs = dict(net_arch=[256, 256])
    model = PPO(
        "MlpPolicy",
        vec_env,
        learning_rate=args.learning_rate,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        n_epochs=args.n_epochs,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,
        policy_kwargs=policy_kwargs,
        verbose=1,
        tensorboard_log="./tron_tensorboard/",
        seed=args.seed,
    )

    print(f"Starting training for {args.total_timesteps} timesteps...")
    model.learn(total_timesteps=args.total_timesteps, progress_bar=True)
    model.save(args.model_path)
    print(f"Model saved to {args.model_path}")

    # Optional evaluation with rendering
    if args.render:
        print("Evaluating trained agent (press Ctrl+C to stop)...")
        obs = vec_env.reset()
        while True:
            action, _states = model.predict(obs, deterministic=True)
            obs, rewards, dones, info = vec_env.step(action)
            env.render()
            if dones[0]:
                obs = vec_env.reset()


if __name__ == "__main__":
    main()