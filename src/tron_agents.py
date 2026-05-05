"""
Learned-agent and LLM-facing adapters for the Tron environment.

This file intentionally stays separate from the core MVC files:
- tron_model.py: fast environment simulation
- tron_controller.py: scripted and keyboard controllers
- tron_view.py: rendering
- tron_app.py: runnable demos / CLI

Action encoding:
    0 = straight, 1 = turn left, 2 = turn right
"""

from __future__ import annotations

from typing import Any, Callable

import numpy as np

from tron_controller import Controller, GreedySpaceController


class AgentAdapter:
    """
    Wrap a learned policy function.

    policy_fn receives:
        observe_lite() -> float32 [envs, players, features]

    policy_fn must return:
        actions -> int [envs, players]
    """

    def __init__(self, policy_fn: Callable[[np.ndarray], np.ndarray]) -> None:
        self.policy_fn = policy_fn

    def actions(self, model: Any) -> np.ndarray:
        return np.asarray(self.policy_fn(model.observe_lite()), dtype=np.int8)


class LLMController:
    """
    Optional LLM-facing controller for a single rendered game.

    This is deliberately NOT for high-throughput training. It converts one env into
    a compact text state, calls a user-supplied llm_fn(prompt) -> str, and parses a
    simple action. Use it for experiments, explanations, or human-vs-agent demos.

    Expected LLM output can be any string containing one of:
        "straight", "left", "right", "0", "1", or "2".
    """

    def __init__(self, llm_fn: Callable[[str], str], player: int = 0, fallback: Controller | None = None) -> None:
        self.llm_fn = llm_fn
        self.player = int(player)
        self.fallback = fallback or GreedySpaceController()

    def actions(self, model: Any) -> np.ndarray:
        acts = self.fallback.actions(model)
        if model.envs != 1:
            raise ValueError("LLMController is intended for envs=1 only")
        prompt = self.make_prompt(model, self.player)
        acts[0, self.player] = self.parse_action(self.llm_fn(prompt))
        return acts

    @staticmethod
    def parse_action(text: str) -> np.int8:
        t = text.strip().lower()
        if "left" in t or t == "1":
            return np.int8(1)
        if "right" in t or t == "2":
            return np.int8(2)
        return np.int8(0)

    @staticmethod
    def make_prompt(model: Any, player: int = 0) -> str:
        legal = model.legal_actions()[0, player]
        x, y = model.pos[0, player]
        heading = int(model.heading[0, player])
        heading_name = ["up", "right", "down", "left"][heading]
        alive = bool(model.alive[0, player])

        others = []
        for p in range(model.players):
            if p == player:
                continue
            ox, oy = model.pos[0, p]
            others.append(f"P{p + 1}=({int(ox)},{int(oy)}), alive={bool(model.alive[0, p])}")

        legal_names = [name for ok, name in zip(legal, ["straight", "left", "right"], strict=False) if ok]
        return (
            "You control a Tron lightcycle. Choose exactly one action: straight, left, or right.\n"
            f"Board: {model.width}x{model.height}.\n"
            f"You are P{player + 1} at ({int(x)},{int(y)}), heading {heading_name}, alive={alive}.\n"
            f"Legal one-step actions: {', '.join(legal_names) or 'none'}.\n"
            f"Other players: {'; '.join(others)}.\n"
            "Reply with only the action."
        )
