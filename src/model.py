"""
Fast model layer for a minimal Tron / Lightcycles environment.

This file has no GUI, keyboard, LLM, or training-framework dependencies.
It is designed for high-throughput batch simulation.

Action encoding:
    0 = straight, 1 = turn left, 2 = turn right
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional

import numpy as np

# Direction encoding: 0 up, 1 right, 2 down, 3 left. Positions are (x, y).
DIR_VECTORS = np.array([[0, -1], [1, 0], [0, 1], [-1, 0]], dtype=np.int16)
TURN = np.array([0, -1, 1], dtype=np.int8)  # action -> heading delta


@dataclass(slots=True)
class StepResult:
    reward: np.ndarray          # float32 [envs, players]
    done: np.ndarray            # bool [envs]
    alive: np.ndarray           # bool [envs, players]
    died: np.ndarray            # bool [envs, players]


@dataclass(slots=True)
class Replay:
    """A small view-only recording of one completed game."""

    width: int
    height: int
    players: int
    owner_frames: list[np.ndarray]
    head_frames: list[np.ndarray]
    alive_frames: list[np.ndarray]

    @classmethod
    def empty(cls, model: "TronBatchModel") -> "Replay":
        return cls(model.width, model.height, model.players, [], [], [])

    def append(self, model: "TronBatchModel", env: int = 0) -> None:
        if model.owner is None:
            owner = model.occupied[env].astype(np.uint8) * 5
        else:
            owner = model.owner[env].copy()
        self.owner_frames.append(owner)
        self.head_frames.append(model.pos[env].copy())
        self.alive_frames.append(model.alive[env].copy())

    def __len__(self) -> int:
        return len(self.owner_frames)


class TronBatchModel:
    """
    Vectorized Lightcycles simulation.

    The hot path is step(): it advances all envs in one NumPy call and stores only
    compact arrays. Rendering data such as owner grids is optional.

    Important arrays:
        occupied: bool [E, H, W]       trail/wall occupancy
        pos:      int16 [E, P, 2]      player head positions, x/y
        heading:  int8 [E, P]          0 up, 1 right, 2 down, 3 left
        alive:    bool [E, P]
        done:     bool [E]

    Memory estimate for occupied only:
        envs * width * height bytes.
        Example: 10_000 envs * 32 * 32 ~= 10 MB.
    """

    def __init__(
        self,
        width: int = 48,
        height: int = 32,
        players: int = 2,
        envs: int = 1,
        max_steps: Optional[int] = None,
        keep_owner: bool = False,
        randomize_spawns: bool = True,
        seed: Optional[int] = None,
    ) -> None:
        if not (2 <= players <= 4):
            raise ValueError("players must be 2, 3, or 4")
        if width < 8 or height < 8:
            raise ValueError("width and height should be at least 8")

        self.width = int(width)
        self.height = int(height)
        self.players = int(players)
        self.envs = int(envs)
        self.max_steps = int(max_steps or (width * height))
        self.keep_owner = bool(keep_owner)
        self.randomize_spawns = bool(randomize_spawns)
        self.rng = np.random.default_rng(seed)

        self.occupied = np.zeros((envs, height, width), dtype=np.bool_)
        self.owner = np.zeros((envs, height, width), dtype=np.uint8) if keep_owner else None
        self.pos = np.zeros((envs, players, 2), dtype=np.int16)
        self.heading = np.zeros((envs, players), dtype=np.int8)
        self.alive = np.ones((envs, players), dtype=np.bool_)
        self.done = np.zeros(envs, dtype=np.bool_)
        self.tick = np.zeros(envs, dtype=np.int32)

        self.reset()

    def reset(self, env_ids: Optional[Iterable[int] | np.ndarray] = None) -> None:
        """Reset all envs, or only env_ids."""
        if env_ids is None:
            ids = np.arange(self.envs)
        else:
            ids = np.asarray(list(env_ids), dtype=np.int64)

        self.occupied[ids] = False
        if self.owner is not None:
            self.owner[ids] = 0
        self.alive[ids] = True
        self.done[ids] = False
        self.tick[ids] = 0

        base_pos, base_heading = self._spawn_layout()
        pos = np.broadcast_to(base_pos[None, :, :], (len(ids), self.players, 2)).copy()

        # Small random jitter improves training diversity while avoiding walls.
        if self.randomize_spawns:
            jitter_x = max(1, self.width // 16)
            jitter_y = max(1, self.height // 16)
            jitter = self.rng.integers(
                low=[-jitter_x, -jitter_y],
                high=[jitter_x + 1, jitter_y + 1],
                size=(len(ids), self.players, 2),
                dtype=np.int16,
            )
            pos += jitter
            pos[..., 0] = np.clip(pos[..., 0], 2, self.width - 3)
            pos[..., 1] = np.clip(pos[..., 1], 2, self.height - 3)

        self.pos[ids] = pos
        self.heading[ids] = base_heading[None, :]

        # Mark starting cells occupied.
        e = ids[:, None]
        p = np.arange(self.players)[None, :]
        x = self.pos[ids, :, 0]
        y = self.pos[ids, :, 1]
        self.occupied[e, y, x] = True
        if self.owner is not None:
            self.owner[e, y, x] = p + 1

    def _spawn_layout(self) -> tuple[np.ndarray, np.ndarray]:
        """Deterministic spawn positions/headings aiming toward the center."""
        w, h, p = self.width, self.height, self.players
        layouts = {
            2: ([(w // 4, h // 2), (3 * w // 4, h // 2)], [1, 3]),
            3: ([(w // 4, h // 2), (3 * w // 4, h // 2), (w // 2, h // 4)], [1, 3, 2]),
            4: (
                [(w // 4, h // 4), (3 * w // 4, 3 * h // 4),
                 (3 * w // 4, h // 4), (w // 4, 3 * h // 4)],
                [1, 3, 2, 0],
            ),
        }
        pos, heading = layouts[p]
        return np.asarray(pos, dtype=np.int16), np.asarray(heading, dtype=np.int8)

    def step(self, actions: np.ndarray) -> StepResult:
        """
        Advance every environment one tick.

        Args:
            actions: int array [envs, players] or [players].
                     0 straight, 1 left, 2 right.
        """
        actions = np.asarray(actions, dtype=np.int8)
        if actions.ndim == 1:
            actions = np.broadcast_to(actions[None, :], (self.envs, self.players))
        if actions.shape != (self.envs, self.players):
            raise ValueError(f"actions must have shape {(self.envs, self.players)}")
        actions = np.clip(actions, 0, 2)

        reward = np.zeros((self.envs, self.players), dtype=np.float32)
        active = (~self.done)[:, None] & self.alive
        if not active.any():
            died = np.zeros((self.envs, self.players), dtype=np.bool_)
            return StepResult(reward, self.done.copy(), self.alive.copy(), died)

        new_heading = (self.heading + TURN[actions]) & 3
        delta = DIR_VECTORS[new_heading]
        new_pos = self.pos + delta
        x = new_pos[..., 0]
        y = new_pos[..., 1]

        in_bounds = (0 <= x) & (x < self.width) & (0 <= y) & (y < self.height)

        # Collision against existing trails/walls. Index only valid cells.
        trail_hit = np.zeros((self.envs, self.players), dtype=np.bool_)
        e_idx = np.broadcast_to(np.arange(self.envs)[:, None], (self.envs, self.players))
        valid = active & in_bounds
        trail_hit[valid] = self.occupied[e_idx[valid], y[valid], x[valid]]

        # Head-to-head collision: multiple live players enter the same empty cell.
        same_x = x[:, :, None] == x[:, None, :]
        same_y = y[:, :, None] == y[:, None, :]
        both_valid = valid[:, :, None] & valid[:, None, :]
        same_cell = same_x & same_y & both_valid
        same_cell &= ~np.eye(self.players, dtype=np.bool_)[None, :, :]
        head_hit = same_cell.any(axis=2)

        died = active & ((~in_bounds) | trail_hit | head_hit)
        survived = active & ~died

        reward[died] = -1.0

        # Apply movement. Dead players keep their previous position, which remains trail.
        self.heading[active] = new_heading[active]
        self.pos[survived] = new_pos[survived]

        # Mark new survivor cells as occupied.
        self.occupied[e_idx[survived], y[survived], x[survived]] = True
        if self.owner is not None:
            p_idx = np.broadcast_to(np.arange(self.players)[None, :], (self.envs, self.players))
            self.owner[e_idx[survived], y[survived], x[survived]] = p_idx[survived] + 1

        self.alive[died] = False
        self.tick[~self.done] += 1

        alive_count = self.alive.sum(axis=1)
        newly_done = (~self.done) & ((alive_count <= 1) | (self.tick >= self.max_steps))

        # Winner reward only on terminal combat states, not on max-step draws.
        has_winner = newly_done & (alive_count == 1)
        if has_winner.any():
            reward[has_winner] += self.alive[has_winner].astype(np.float32)

        self.done[newly_done] = True
        return StepResult(reward, self.done.copy(), self.alive.copy(), died)

    def legal_actions(self) -> np.ndarray:
        """Return bool [envs, players, 3] for one-step safe actions."""
        candidate_heading = (self.heading[:, :, None] + TURN[None, None, :]) & 3
        candidate_delta = DIR_VECTORS[candidate_heading]
        candidate_pos = self.pos[:, :, None, :] + candidate_delta
        x = candidate_pos[..., 0]
        y = candidate_pos[..., 1]
        in_bounds = (0 <= x) & (x < self.width) & (0 <= y) & (y < self.height)

        free = np.zeros((self.envs, self.players, 3), dtype=np.bool_)
        e = np.broadcast_to(np.arange(self.envs)[:, None, None], (self.envs, self.players, 3))
        valid = in_bounds & self.alive[:, :, None] & (~self.done[:, None, None])
        free[valid] = ~self.occupied[e[valid], y[valid], x[valid]]
        return free

    def observe_lite(self) -> np.ndarray:
        """
        Compact float observation [envs, players, features].

        Features per player:
            legal_straight/left/right (3)
            x_norm, y_norm (2)
            heading one-hot (4)
            alive (1)
            for each other player: rel_x, rel_y, alive (3 * (players - 1))
        """
        legal = self.legal_actions().astype(np.float32)
        xy = self.pos.astype(np.float32)
        xy[..., 0] /= max(1, self.width - 1)
        xy[..., 1] /= max(1, self.height - 1)
        heading_oh = np.eye(4, dtype=np.float32)[self.heading]
        alive = self.alive[..., None].astype(np.float32)

        rel_parts = []
        for i in range(self.players):
            parts = []
            for j in range(self.players):
                if i == j:
                    continue
                dxy = (self.pos[:, j] - self.pos[:, i]).astype(np.float32)
                dxy[:, 0] /= max(1, self.width - 1)
                dxy[:, 1] /= max(1, self.height - 1)
                parts.append(np.concatenate([dxy, self.alive[:, j:j + 1].astype(np.float32)], axis=1))
            rel_parts.append(np.concatenate(parts, axis=1))
        rel = np.stack(rel_parts, axis=1)
        return np.concatenate([legal, xy, heading_oh, alive, rel], axis=2)

    def observe_grid(self) -> np.ndarray:
        """
        CNN-friendly observation [envs, 1 + players, height, width].

        Channel 0 is occupied cells. Channels 1..players are player heads.
        This can be memory-heavy for large batches.
        """
        obs = np.zeros((self.envs, 1 + self.players, self.height, self.width), dtype=np.float32)
        obs[:, 0] = self.occupied
        e = np.arange(self.envs)[:, None]
        p = np.arange(self.players)[None, :]
        x = self.pos[..., 0]
        y = self.pos[..., 1]
        obs[e, p + 1, y, x] = self.alive.astype(np.float32)
        return obs

    def auto_reset_done(self) -> np.ndarray:
        """Reset terminal envs and return their ids."""
        ids = np.flatnonzero(self.done)
        if len(ids):
            self.reset(ids)
        return ids
