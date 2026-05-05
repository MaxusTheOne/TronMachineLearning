"""
Runnable app / demo layer for the modular Tron project.

Run examples:
    python tron_app.py --mode watch --players 4
    python tron_app.py --mode human --players 2
    python tron_app.py --mode replay --players 4
    python tron_app.py --mode benchmark --envs 10000 --steps 1000
    python tron_app.py --mode train-skeleton

Dependencies:
    pip install numpy

The GUI uses tkinter from the Python standard library, not pygame.
"""

from __future__ import annotations

import argparse
import time
from typing import Optional

from tron_controller import Controller, GreedySpaceController, KeyStateController, RandomController
from tron_model import Replay, TronBatchModel
from tron_view import GameView


def play_live(
    players: int = 2,
    human: bool = False,
    width: int = 96,
    height: int = 64,
    scale: int = 16,
    fps: int = 10,
    seed: Optional[int] = None,
) -> None:
    model = TronBatchModel(
        width=width,
        height=height,
        players=players,
        envs=1,
        keep_owner=True,
        seed=seed,
    )
    view = GameView(model, scale=scale, fps=fps)
    controller: Controller = KeyStateController(view.pressed, human_players={0} if human else set(), fallback=GreedySpaceController())

    try:
        while view.poll():
            if view.take_restart_request():
                model.reset()
            if not model.done[0]:
                model.step(controller.actions(model))
            view.render(model)
    finally:
        view.close()


def record_game(
    controller: Controller,
    players: int = 2,
    width: int = 48,
    height: int = 32,
    seed: Optional[int] = None,
) -> Replay:
    """Run one full game without rendering, then return a replay object."""
    model = TronBatchModel(width=width, height=height, players=players, envs=1, keep_owner=True, seed=seed)
    replay = Replay.empty(model)
    replay.append(model)
    while not model.done[0]:
        model.step(controller.actions(model))
        replay.append(model)
    return replay


def play_replay(replay: Replay, scale: int = 16, fps: int = 20) -> None:
    """Display a completed game. Space pauses, arrows scrub, R restarts replay."""
    view = GameView(replay.width, replay.height, scale=scale, fps=fps)
    frame = 0
    paused = False
    try:
        while view.poll():
            if view.take_restart_request():
                frame = 0
                paused = False
            if "space" in view.just_pressed:
                paused = not paused
            if "left_arrow" in view.just_pressed:
                frame = max(0, frame - 5)
            if "right_arrow" in view.just_pressed:
                frame = min(len(replay) - 1, frame + 5)

            view.render_replay(replay, frame)
            if not paused:
                frame = min(len(replay) - 1, frame + 1)
    finally:
        view.close()


def benchmark(envs: int = 10_000, steps: int = 1_000, players: int = 2) -> None:
    model = TronBatchModel(width=32, height=32, players=players, envs=envs, keep_owner=False)
    ctrl = RandomController(0)
    t0 = time.perf_counter()
    transitions = 0
    for _ in range(steps):
        model.step(ctrl.actions(model))
        transitions += envs
        model.auto_reset_done()
    dt = max(1e-9, time.perf_counter() - t0)
    print(f"{transitions / dt:,.0f} env-steps/sec  ({envs=} {steps=} {players=})")


def train_loop_skeleton() -> None:
    """Tiny example of how a trainer would use the model without rendering."""
    env = TronBatchModel(width=32, height=32, players=2, envs=4096, keep_owner=False)
    policy = RandomController(0)  # Replace with tron_agents.AgentAdapter(your_policy_fn).

    for update in range(10):
        obs = env.observe_lite()
        actions = policy.actions(env)
        result = env.step(actions)
        next_obs = env.observe_lite()

        # Store obs/actions/result.reward/result.done/next_obs in your replay buffer here.
        _ = (obs, actions, result, next_obs)
        env.auto_reset_done()
        print(f"update {update}: mean reward {result.reward.mean():+.4f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Minimal MVC Tron / Lightcycles")
    parser.add_argument("--mode", choices=["watch", "human", "replay", "benchmark", "train-skeleton"], default="watch")
    parser.add_argument("--players", type=int, default=2, choices=[2, 3, 4])
    parser.add_argument("--width", type=int, default=48)
    parser.add_argument("--height", type=int, default=32)
    parser.add_argument("--scale", type=int, default=16)
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--envs", type=int, default=10_000)
    parser.add_argument("--steps", type=int, default=1_000)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    if args.mode == "benchmark":
        benchmark(envs=args.envs, steps=args.steps, players=args.players)
    elif args.mode == "train-skeleton":
        train_loop_skeleton()
    elif args.mode == "replay":
        replay = record_game(GreedySpaceController(), players=args.players, width=args.width, height=args.height, seed=args.seed)
        play_replay(replay, scale=args.scale, fps=args.fps)
    else:
        play_live(
            players=args.players,
            human=args.mode == "human",
            width=args.width,
            height=args.height,
            scale=args.scale,
            fps=args.fps,
            seed=args.seed,
        )


if __name__ == "__main__":
    main()
