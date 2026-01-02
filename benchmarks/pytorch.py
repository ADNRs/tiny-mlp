import torch
import time
import numpy as np
import sys

if len(sys.argv) > 1 and sys.argv[1] == '--single-thread':
    torch.set_num_threads(1)

def benchmark_median(func, warmup=5, iterations=100):
    for _ in range(warmup):
        func()

    times = []
    for _ in range(iterations):
        start = time.perf_counter()
        func()
        end = time.perf_counter()
        times.append(end - start)

    return np.median(times)

def run_benchmark(M, K, N):
    print(f"\n=== Matrix Multiply Benchmark ===")
    print(f"Matrix sizes: A({M}x{K}) * B({K}x{N}) = C({M}x{N})")

    flops = 2.0 * M * K * N
    print(f"Operations per matmul: {flops:.2e} FLOPs")

    A = torch.randn(M, K)
    B = torch.randn(K, N)

    def matmul():
        return torch.matmul(A, B)

    time_ms = benchmark_median(matmul) * 1000
    gflops = (flops / (time_ms / 1000)) / 1e9

    print(f"\n--- PyTorch Performance ---")
    print(f"Time: {time_ms:.2f} ms (median of 100 runs)")
    print(f"Performance: {gflops:.2f} GFLOPS")

if __name__ == "__main__":
    print("PyTorch CPU Matrix Multiplication Benchmark")
    print("===========================================")
    print(f"PyTorch version: {torch.__version__}")
    print(f"Number of threads: {torch.get_num_threads()}")
    print("(All times are median of 100 iterations)")

    run_benchmark(18, 96, 32)
    run_benchmark(24, 96, 64)
    run_benchmark(24, 256, 64)
    run_benchmark(48, 128, 64)
    run_benchmark(192, 128, 64)
    run_benchmark(192, 128, 128)
    run_benchmark(480, 16, 512)
    run_benchmark(192, 256, 256)
    run_benchmark(384, 256, 256)
    run_benchmark(480, 256, 512)
    run_benchmark(1024, 1024, 1024)
    run_benchmark(1020, 1152, 1152)
    run_benchmark(1920, 2304, 2304)
    run_benchmark(2304, 2560, 2304)
