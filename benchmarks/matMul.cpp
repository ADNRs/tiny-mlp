#include "TinyMLP.hpp"
#include <algorithm>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <memory>
#include <vector>

using namespace tinymlp;

template <typename Func>
double benchmarkMedian(Func &&F, unsigned int Warmup = 5,
                       unsigned int Iterations = 100) {
  std::vector<double> Times;
  Times.reserve(Iterations);

  for (unsigned int I = 0; I < Warmup; I++)
    F();

  for (unsigned int I = 0; I < Iterations; I++) {
    auto start = std::chrono::high_resolution_clock::now();
    F();
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff = end - start;
    Times.push_back(diff.count());
  }

  std::sort(Times.begin(), Times.end());
  size_t Mid = Times.size() / 2;
  if (Times.size() % 2 == 0)
    return (Times[Mid - 1] + Times[Mid]) / 2.0;
  else
    return Times[Mid];
}

template <size_t M, size_t K, size_t N, typename T = float>
void runBenchmark() {
  std::cout << "\n=== Matrix Multiply Benchmark ===" << std::endl;
  std::cout << "Matrix sizes: A(" << M << "x" << K << ") * B(" << K << "x" << N
            << ") = C(" << M << "x" << N << ")" << std::endl;

  const double flops = 2.0 * M * K * N;
  std::cout << "Operations per matmul: " << flops / 1e9 << " GFLOPs"
            << std::endl;

  auto A = std::make_unique<Mat<M, K, T>>();
  auto B = std::make_unique<Mat<K, N, T>>();
  auto C = std::make_unique<Mat<M, N, T>>();
  auto CRef = std::make_unique<Mat<M, N, T>>();

  for (size_t I = 0; I < M; I++)
    for (size_t X = 0; X < K; X++)
      (*A)(I, X) = static_cast<T>(rand()) / static_cast<T>(RAND_MAX);

  for (size_t X = 0; X < K; X++)
    for (size_t J = 0; J < N; J++)
      (*B)(X, J) = static_cast<T>(rand()) / static_cast<T>(RAND_MAX);

  matFill(*CRef, static_cast<T>(0.0));

  for (size_t I = 0; I < M; I++) {
    for (size_t X = 0; X < K; X++) {
      T ValA = (*A)(I, X);
      for (size_t J = 0; J < N; J++)
        (*CRef)(I, J) += ValA * (*B)(X, J);
    }
  }

  auto TimeMatMul = benchmarkMedian([&]() {
    matFill(*C, static_cast<T>(0.0));
    matMul(*A, *B, *C);
  });

  double GFLOPs = (flops / TimeMatMul) / 1e9;

  std::cout << "--- matMul Performance ---" << std::endl;
  std::cout << "Time: " << TimeMatMul * 1000 << " ms (median of 100 runs)"
            << std::endl;
  std::cout << "Performance: " << std::fixed << std::setprecision(2) << GFLOPs
            << " GFLOPS" << std::endl;

  double MaxDiff = 0.0;
  double MaxRelErr = 0.0;

  for (size_t I = 0; I < M; I++) {
    for (size_t J = 0; J < N; J++) {
      double Diff = std::abs((*C)(I, J) - (*CRef)(I, J));
      MaxDiff = std::max(MaxDiff, Diff);
      if (std::abs((*CRef)(I, J)) > 1e-30) {
        double RelErr = Diff / std::abs((*CRef)(I, J));
        MaxRelErr = std::max(MaxRelErr, RelErr);
      }
    }
  }

  if (MaxRelErr < 1e-30)
    std::cout << "✓ Results are correct!" << std::endl;
  else
    std::cout << "✗ Results differ significantly!" << std::endl;
}

int main() {
  srand(42);

  std::cout << "TinyMLP Matrix Multiplication Benchmark" << std::endl;
  std::cout << "=======================================" << std::endl;
  std::cout << "(All times are median of 100 iterations)" << std::endl;

  runBenchmark<18, 96, 32>();
  runBenchmark<24, 96, 64>();
  runBenchmark<24, 256, 64>();
  runBenchmark<48, 128, 64>();
  runBenchmark<192, 128, 64>();
  runBenchmark<192, 128, 128>();
  runBenchmark<480, 16, 512>();
  runBenchmark<192, 256, 256>();
  runBenchmark<384, 256, 256>();
  runBenchmark<480, 256, 512>();
  runBenchmark<1024, 1024, 1024>();
  runBenchmark<1020, 1152, 1152>();
  runBenchmark<1920, 2304, 2304>();
  runBenchmark<2304, 2560, 2304>();

  return 0;
}
