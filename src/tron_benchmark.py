import time
import cProfile
import pstats
import numpy as np
from memory_profiler import profile

from src.tron_model import TronBatchModel


def benchmark_performance():
    """Benchmark your TronBatchModel"""
    configs = [
        (1, 2, 32, 32),  # Single game
        (100, 2, 32, 32),  # 100 games, 2 players
        (1000, 2, 32, 32),  # 1000 games
        (100, 4, 64, 64),  # 4 players, larger map
    ]

    for envs, players, w, h in configs:
        model = TronBatchModel(width=w, height=h, players=players,
                               envs=envs, keep_owner=False)

        # Warmup
        actions = np.random.randint(0, 3, (envs, players))
        for _ in range(10):
            model.step(actions)

        # Benchmark
        start = time.perf_counter()
        steps = 1000
        for _ in range(steps):
            model.step(actions)
        elapsed = time.perf_counter() - start

        print(f"{envs:4d} envs, {players} players: "
              f"{steps / elapsed:.0f} steps/sec, "
              f"{(envs * players * steps) / elapsed:.0f} agent-steps/sec")


def profile_hot_path():
    """Profile the step() method"""
    model = TronBatchModel(envs=100, players=2)
    actions = np.random.randint(0, 3, (100, 2))

    profiler = cProfile.Profile()
    profiler.enable()

    for _ in range(1000):
        model.step(actions)

    profiler.disable()
    stats = pstats.Stats(profiler)
    stats.sort_stats('cumulative')
    stats.print_stats(20)  # Top 20 functions


@profile
def memory_profile():
    """Profile memory usage"""
    model = TronBatchModel(envs=1000, players=4, width=64, height=64)
    model.observe_grid()  # Check memory for CNN observations
    model.observe_lite()  # Check memory for lite observations

    actions = np.random.randint(0, 3, (1000, 4))
    for _ in range(100):
        model.step(actions)


if __name__ == "__main__":
    benchmark_performance()
    # profile_hot_path()  # Uncomment for detailed profiling
    # memory_profile()