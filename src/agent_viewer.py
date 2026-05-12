#!/usr/bin/env python3
"""
Visualize a trained RL agent (from stable-baselines3) playing Tron.

This script loads a saved PPO model and lets it control player 0 while
opponents follow a simple heuristic (greedy or random). The game is rendered
using the existing view.py and controller.py modules.

Usage:
    python play_trained_agent.py --model_path tron_ppo.zip --players 2 --opponent greedy
"""

import argparse
import time
from typing import Optional

import numpy as np
from stable_baselines3 import PPO

# Import the core game model and visualization components
from model import TronBatchModel
from view import GameView
from controller import GreedySpaceController, RandomController


class TrainedAgentController:
    """
    Controller that uses a pretrained Stable-Baselines3 model to choose actions
    for player 0. The model must have been trained on the TronSingleAgentEnv,
    which uses the 'lite' observation for a single agent.
    """

    def __init__(self, model_path: str, device: str = "auto"):
        self.model = PPO.load(model_path, device=device)

    def action(self, obs: np.ndarray) -> int:
        """
        Given the observation for player 0 (shape: (feat_dim,)),
        return the chosen action (0=straight, 1=left, 2=right).
        """
        # The predict method expects a batch dimension
        action, _ = self.model.predict(obs, deterministic=True)
        return int(action)


def main():
    parser = argparse.ArgumentParser(description="Watch a trained Tron agent play.")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the saved PPO model (.zip)")
    parser.add_argument("--width", type=int, default=48, help="Board width")
    parser.add_argument("--height", type=int, default=32, help="Board height")
    parser.add_argument("--players", type=int, default=2, help="Total players (agent + opponents)")
    parser.add_argument("--opponent", type=str, default="greedy", choices=["greedy", "random"],
                        help="Policy for opponents (player 1, 2, ...)")
    parser.add_argument("--fps", type=int, default=30, help="Rendering frames per second")
    parser.add_argument("--scale", type=int, default=16, help="Cell size in pixels")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    args = parser.parse_args()

    # Create the game model with owner tracking (required for proper colours)
    model = TronBatchModel(
        width=args.width,
        height=args.height,
        players=args.players,
        envs=1,
        keep_owner=True,
        randomize_spawns=True,
        seed=args.seed,
    )

    # Set up opponent controller
    if args.opponent == "greedy":
        opponent_ctrl = GreedySpaceController()
    else:
        opponent_ctrl = RandomController(seed=args.seed)

    # Load the trained agent controller
    agent_ctrl = TrainedAgentController(args.model_path)

    # Create the tkinter view
    view = GameView(model, scale=args.scale, fps=args.fps)

    print("Starting game. Close the window to exit.")
    print("Press R or click 'Restart game' to reset the match.")

    # Helper to get the agent's observation from the model
    def get_agent_obs() -> np.ndarray:
        # observe_lite() returns [envs, players, features], take env 0, player 0
        return model.observe_lite()[0, 0]

    # Main game loop
    while view.poll():
        # Check for restart request
        if view.take_restart_request():
            model.reset(env_ids=[0])
            # Also reset opponent controller if it has internal state
            if isinstance(opponent_ctrl, GreedySpaceController):
                # GreedySpaceController is stateless, nothing to do
                pass
            # Reset agent controller (no internal state)
            continue

        # If the episode is already done, wait for restart (the view shows a banner)
        if model.done[0]:
            view.render(model)
            # Small sleep to avoid busy-looping while waiting for restart
            time.sleep(0.05)
            continue

        # Get observation and action for the agent (player 0)
        obs = get_agent_obs()
        agent_action = agent_ctrl.action(obs)

        # Get actions for all players:
        # - player 0: from the trained agent
        # - players 1..P-1: from the opponent controller
        actions = np.zeros((1, args.players), dtype=np.int8)
        actions[0, 0] = agent_action

        # Opponent actions: opponent_ctrl.actions(model) returns (envs, players) array
        # We need only players 1.. for this single environment
        all_actions = opponent_ctrl.actions(model)   # shape (1, players)
        actions[0, 1:] = all_actions[0, 1:]

        # Step the environment
        result = model.step(actions)

        # Render the current state
        view.render(model)

        # Optional: small delay to respect FPS (view.sleep_frame already called inside render)
        # The view handles its own timer via after() calls, so we don't need additional sleep.

    print("Window closed.")


if __name__ == "__main__":
    main()