#!/usr/bin/env python3
"""
Performance benchmark for TronBatchModel.
Compares original vs. optimized version.
"""

import time
import numpy as np

# Import the optimized model
from batch_model import TronBatchModel


def benchmark(model, steps=1000, verbose=False):
    """Run a benchmark and return steps per second."""
    envs, players = model.envs, model.players
    actions = np.random.randint(0, 3, (envs, players), dtype=np.int8)

    # Warmup
    for _ in range(10):
        result = model.step(actions)

    # Benchmark
    start = time.perf_counter()
    for _ in range(steps):
        result = model.step(actions)
    elapsed = time.perf_counter() - start

    step_rate = steps / elapsed
    agent_rate = (envs * players * steps) / elapsed

    if verbose:
        print(f"  Steps: {steps}, Time: {elapsed:.3f}s")
        print(f"  Step rate: {step_rate:.0f} steps/sec")
        print(f"  Agent rate: {agent_rate:.0f} agent-steps/sec")

    return step_rate, agent_rate


def run_benchmarks():
    configs = [
        (1, 2, 32, 32, "Single game"),
        (100, 2, 32, 32, "100 games, 2 players"),
        (500, 2, 32, 32, "500 games, 2 players"),
        (100, 4, 48, 48, "100 games, 4 players"),
        (200, 4, 48, 48, "200 games, 4 players"),
    ]

    print("=" * 80)
    print(f"{'Config':<40} {'Steps/sec':>12} {'Agent-steps/sec':>18} {'Numba':>8}")
    print("=" * 80)

    # Test with Numba enabled
    for envs, players, w, h, desc in configs:
        try:
            model = TronBatchModel(
                width=w, height=h, players=players,
                envs=envs, keep_owner=False, use_numba=True
            )
            step_rate, agent_rate = benchmark(model)
            print(f"{desc:<40} {step_rate:12.0f} {agent_rate:18.0f} {'Yes':>8}")
        except Exception as e:
            print(f"{desc:<40} {'ERROR':>12} {'':18} {'Yes':>8}")
            print(f"  Error: {e}")

    # Test with Numba disabled for comparison (only one config to save time)
    print("\n" + "-" * 80)
    print("Comparison with Numba disabled:")
    print("-" * 80)

    try:
        model = TronBatchModel(
            width=48, height=48, players=2, envs=200,
            keep_owner=False, use_numba=False
        )
        step_rate, agent_rate = benchmark(model)
        print(f"200 games, 2 players (no Numba): {step_rate:12.0f} steps/sec, {agent_rate:18.0f} agent-steps/sec")
    except Exception as e:
        print(f"Error with Numba disabled: {e}")


def profile_memory():
    """Profile memory usage of the model."""
    import sys

    print("\n" + "=" * 80)
    print("Memory Usage Profile")
    print("=" * 80)

    configs = [
        (100, 2, 32, 32),
        (500, 2, 32, 32),
        (1000, 2, 32, 32),
        (100, 4, 48, 48),
        (500, 4, 48, 48),
    ]

    for envs, players, w, h in configs:
        model = TronBatchModel(width=w, height=h, players=players, envs=envs, keep_owner=False)

        # Calculate approximate memory usage
        occupied_mem = envs * h * w * np.dtype(bool).itemsize / (1024 * 1024)
        pos_mem = envs * players * 2 * np.dtype(np.int16).itemsize / (1024 * 1024)
        heading_mem = envs * players * np.dtype(np.int8).itemsize / (1024 * 1024)
        alive_mem = envs * players * np.dtype(bool).itemsize / (1024 * 1024)
        done_mem = envs * np.dtype(bool).itemsize / (1024 * 1024)
        tick_mem = envs * np.dtype(np.int32).itemsize / (1024 * 1024)

        total_mem = occupied_mem + pos_mem + heading_mem + alive_mem + done_mem + tick_mem

        print(f"{envs:4d} envs, {players} players, {w}x{h}:")
        print(f"  Occupied: {occupied_mem:.2f} MB")
        print(f"  Positions: {pos_mem:.2f} MB")
        print(f"  Total: {total_mem:.2f} MB")
        print()


def quick_test():
    """Quick test to ensure model works correctly."""
    print("\n" + "=" * 80)
    print("Quick Functionality Test")
    print("=" * 80)

    model = TronBatchModel(width=16, height=16, players=2, envs=3, use_numba=True)

    actions = np.array([
        [0, 1],  # env0: straight, right
        [1, 2],  # env1: left, right
        [2, 0]  # env2: right, straight
    ], dtype=np.int8)

    print("Initial state:")
    print(f"  Positions: {model.pos}")
    print(f"  Alive: {model.alive}")

    for step in range(5):
        result = model.step(actions)
        print(f"\nStep {step + 1}:")
        print(f"  Rewards: {result.reward}")
        print(f"  Died: {result.died}")
        print(f"  Alive: {result.alive}")
        print(f"  Done: {result.done}")

    print("\nTest passed!")


if __name__ == "__main__":
    # Quick test first
    quick_test()

    # Run benchmarks
    run_benchmarks()

    # Show memory profile
    profile_memory()