import os
import time
import psutil
import gc


def _time_memory_benchmark(func, *args, **kwargs) -> None:
    """Benchmark the execution time and memory usage of a function."""
    gc.collect()

    process = psutil.Process(os.getpid())
    mem_before = process.memory_info().rss / (1024 * 1024)

    start_time = time.perf_counter()
    func(*args, **kwargs)
    end_time = time.perf_counter()

    time.sleep(0.1)
    mem_after = process.memory_info().rss / (1024 * 1024)
    mem_delta = mem_after - mem_before

    print(f"⏱️ Time:          {end_time - start_time:.2f}s")
    print(f"📈 Memory Delta:  {mem_delta:.2f} MB")
    print(f"🏠 Total RSS:     {mem_after:.2f} MB")
