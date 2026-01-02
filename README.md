# TinyMLP: A High-Performance, Header-Only C++17 Deep Learning Library

TinyMLP is a lightweight, zero-dependency, platform-independent C++17 header-only library designed for training and inference of Multi-Layer Perceptrons (MLPs) on CPU. It leverages template metaprogramming to achieve zero-overhead abstractions, ensuring that high-level model definitions compile down to highly optimized, flat machine code.

## Installation

Just include the header `TinyMLP.hpp`.

```C++
#include "TinyMLP.hpp"
```

## Usage

### Defining a three-layer MLP

```C++
struct Params {
  using DataType = float;
  static constexpr size_t FeatDim = 4;
  static constexpr size_t HiddenDim = 32;
  static constexpr size_t LabelDim = 3;
};

template <size_t BatchSize>
using Layers = tinymlp::LayerList<
    tinymlp::Linear<BatchSize, Params::FeatDim, Params::HiddenDim,
                    Params::DataType>,
    tinymlp::Tanh<BatchSize, Params::HiddenDim, Params::DataType>,
    tinymlp::Dropout<BatchSize, Params::HiddenDim, Params::DataType>,
    tinymlp::Linear<BatchSize, Params::HiddenDim, Params::LabelDim,
                    Params::DataType>>;
```

### Creating an inference wrapper

```C++
template <size_t BatchSize>
using Model =
    tinymlp::Sequential<BatchSize, Layers<BatchSize>, Params::DataType>;
```

### Creating a training wrapper for classification

```C++
template <size_t BatchSize>
using ModelTrainer = tinymlp::SequentialTrainer<
    BatchSize,
    tinymlp::CrossEntropy<BatchSize, Params::LabelDim, Params::DataType>,
    Layers<BatchSize>, Params::DataType>;
```

### Using `Dataset`

```C++
auto X = std::make_unique<Mat<NumData, FeatDim, Params::DataType>>();
auto Y = std::make_unique<Mat<NumData, LabelDim, Params::DataType>>();

parseData(DATA_PATH, *X, *Y);
tinymlp::shuffle(*X, *Y);

auto Dataset =
    std::make_unique<tinymlp::Dataset<NumData, FeatDim, LabelDim, BatchSize,
                                      Params::DataType>>(*X, *Y);
```

### Training

```C++
auto MT = std::make_unique<ModelTrainer<BatchSize>>();

for (unsigned int I = 0; I < NumEpochs; I++) {
  Params::DataType Loss = 0;

  for (const auto &[X, Y] : Dataset->epoch()) {
    Params::DataType LocalLoss = 0;
    MT->clearGrad();
    MT->propagate(X, Y, LocalLoss);
    MT->update();
    Loss += LocalLoss;
  }

  std::cout << "Epoch: " << I + 1 << ", Loss:" << Loss / Dataset->size()
            << "\n";
}
```

### Saving model weights

```C++
std::ofstream Out(WEIGHTS_PATH, std::ios::binary);
MT->save(Out);
Out.close();
```

### Load model weights

```C++
std::ifstream In(WEIGHTS_PATH, std::ios::binary);
auto M = std::make_unique<Model<1>>();
M->load(In);
In.close();
```

### Predicting the label of the first sample

```C++
auto Pred =
    std::make_unique<tinymlp::Mat<1, Params::LabelDim, Params::DataType>>();
auto PredLabel = std::make_unique<tinymlp::Mat<1, 1, Params::DataType>>();

M->forward(X->template getView<1>(), *Pred);
tinymlp::matArgmaxRow(*Pred, *PredLabel);
std::cout << "Prediction: " << PredLabel(0, 0) << std::endl;
```

You can find more examples in `examples/`.

## Design

To maximize compiler optimization, TinyMLP computes everything possible at compile time, including matrix shape checking and the computation graph. Also, users have to manually manage the memory allocations. TinyMLP only allocate necessary memory statically at compile time. These design choices make the library inherently less flexible than widely-used frameworks like PyTorch or TensorFlow. Users must manually implement the forward and backward computation graphs for any unsupported layer and any architecture other than a simple sequential model. TinyMLP is built primarily for fast MLP model inference while still providing a streamlined training interface for C++ users.

**TL;DR**: Due to the heavy reliance on template metaprogramming, users should be prepared for frightening compiler error messages.

## Available Components

+ **Layer**
  + `Linear`
  + `Dropout`
+ **Activation**
  + `ReLU`
  + `Sigmoid`
  + `Tanh`
+ **Loss**
  + `BCEWithLogits`
  + `MSE`
  + `CrossEntropy`
+ **Optimizer**
  + `NAdamW` (Nesterov-accelerated AdamW)
+ **Utility**
  + `Dataset` (for iterating mini-batches)
  + `Sequential` (wrapper that provides easy-to-use methods for inference)
  + `SequentialTrainer` (wrapper that provides easy-to-use methods for both training and inference)
  + `shuffle` (shuffling data and labels simultaneously)
  + `smoothLabel`

## Performance

TinyMLP's single-threaded matrix multiplication competes with optimized BLAS implementations by leveraging cache-aware tiling and compiler auto-vectorization. This implementation uses neither intrinsic functions nor inline assembly, but is carefully structured to guide compilers to emit efficient SIMD instructions (such as AVX2 and NEON) through explicit data alignment and register-level blocking.

### Hint

Depending on your target ISA and CPU cache hierarchy, you may want to adjust the default parameters that can be found at the beginning of `TinyMLP.hpp`. These constants control the register blocking and cache tiling strategies, which directly affect the SIMD instructions emitted by the compiler. You can modify and run `benchmarks/matMul.cpp` to profile specific cases and verify the optimal settings for your hardware.

```C++
static constexpr size_t Align = 64;
static constexpr size_t RegTM = 4;
static constexpr size_t RegTN = 16;

struct L1Info {
  static constexpr size_t Size = 32 * 1024;
  static constexpr size_t Assoc = 12;
  static constexpr size_t LineSize = 64;
};

struct L2Info {
  static constexpr size_t Size = 256 * 1024;
  static constexpr size_t Assoc = 8;
  static constexpr size_t LineSize = 64;
};
```

### Single-Precision Matrix Multiplication

#### Shared Settings

*GFLOPS is short for giga floating-point operations per second; each reported value is the median of the calculated GFLOPS from 100 repeated trails; testing scripts can be found in `benchmarks/`; the compilation flags of TinyMLP are `-std=c++17 -O3 -march=native -mno-avx512f -fno-math-errno -fno-trapping-math -ffp-contract=fast -funroll-loops -fconstexpr-steps=100000000`.*

#### Intel Core i5-11500

+ **Theoretical GFLOPS**
  + 4.6 GHz $\times$ 8 floats $\times$ 2 FMA units $\times$ 2 ops (fmul + fadd) $=$ 147.2 GFLOPS
+ **Operating System**
  + Ubuntu 22.04.1
+ **Comparison**
  + TinyMLP (Clang) vs PyTorch (Intel oneMKL)

##### Single-Threaded

| Matrix Size (M, K, N) | TinyMLP (GFLOPS) | PyTorch (GFLOPS) | Speedup |
| --------------------- | ---------------- | ---------------- | ------- |
| 18×96×32              | **109.82**       | 33.91            | 323.86% |
| 24×96×64              | **115.95**       | 63.29            | 183.20% |
| 24×256×64             | **95.81**        | 94.62            | 101.26% |
| 48×128×64             | **120.56**       | 94.86            | 127.09% |
| 192×128×64            | 118.94           | **122.03**       | 97.47%  |
| 192×128×128           | 122.73           | **127.86**       | 95.99%  |
| 480×16×512            | 88.35            | **123.61**       | 71.47%  |
| 192×256×256           | 119.09           | **125.50**       | 94.89%  |
| 384×256×256           | 117.17           | **128.36**       | 91.28%  |
| 480×256×512           | 119.15           | **131.52**       | 90.60%  |
| 1024×1024×1024        | 118.10           | **131.68**       | 89.68%  |
| 1020×1152×1152        | 119.58           | **131.90**       | 90.66%  |
| 1920×2304×2304        | 114.23           | **133.50**       | 85.57%  |
| 2304×2560×2304        | 112.32           | **133.67**       | 84.03%  |

##### Multi-Threaded

| Matrix Size (M, K, N) | TinyMLP (GFLOPS) | PyTorch (GFLOPS) | Speedup |
| --------------------- | ---------------- | ---------------- | ------- |
| 18×96×32              | **110.48**       | 33.71            | 327.74% |
| 24×96×64              | **94.28**        | 63.65            | 148.12% |
| 24×256×64             | **123.02**       | 80.62            | 152.59% |
| 48×128×64             | 120.84           | **157.95**       | 76.68%  |
| 192×128×64            | 119.84           | **418.78**       | 28.61%  |
| 192×128×128           | 122.95           | **546.16**       | 22.51%  |
| 480×16×512            | 90.30            | **284.08**       | 31.79%  |
| 192×256×256           | 120.33           | **350.41**       | 34.34%  |
| 384×256×256           | 119.69           | **667.03**       | 17.94%  |
| 480×256×512           | 119.04           | **530.27**       | 22.45%  |
| 1024×1024×1024        | 483.39           | **712.70**       | 67.83%  |
| 1020×1152×1152        | 504.01           | **718.86**       | 70.11%  |
| 1920×2304×2304        | 490.31           | **706.16**       | 69.43%  |
| 2304×2560×2304        | 538.12           | **688.34**       | 78.18%  |

*The Clang version is 21.1.7; the PyTorch version is 2.9.1+cpu; the oneMKL version is 2024.2-Product Build 20240605.*

#### Intel Core Ultra 7 258V

+ **Theoretical GFLOPS**
  + **P-Core**: 4.8 GHz $\times$ 8 floats $\times$ 2 FMA units $\times$ 2 ops (fmul + fadd) $=$ 153.6 GFLOPS
  + **E-Core**: 3.7 GHz $\times$ 4 floats $\times$ 4 FMA units $\times$ 2 ops (fmul + fadd) $=$ 118.4 GFLOPS
+ **Operating System**
  + Ubuntu 24.04.2
+ **Comparison**
  + TinyMLP (Clang) vs PyTorch (Intel oneMKL)

##### Single-Threaded

| Matrix Size (M, K, N) | TinyMLP (GFLOPS) | PyTorch (GFLOPS) | Speedup |
| --------------------- | ---------------- | ---------------- | ------- |
| 18×96×32              | **119.04**       | 56.47            | 210.80% |
| 24×96×64              | **119.88**       | 82.03            | 146.14% |
| 24×256×64             | **130.80**       | 115.19           | 113.55% |
| 48×128×64             | **128.01**       | 115.41           | 110.92% |
| 192×128×64            | **131.35**       | 130.31           | 100.80% |
| 192×128×128           | **135.17**       | 119.12           | 113.47% |
| 480×16×512            | 107.77           | **128.37**       | 83.95%  |
| 192×256×256           | **130.46**       | 124.12           | 105.11% |
| 384×256×256           | **130.50**       | 122.84           | 106.24% |
| 480×256×512           | **129.53**       | 118.95           | 108.89% |
| 1024×1024×1024        | **112.19**       | 111.33           | 100.77% |
| 1020×1152×1152        | **110.53**       | 109.74           | 100.72% |
| 1920×2304×2304        | 108.21           | **111.02**       | 97.47%  |
| 2304×2560×2304        | 108.60           | **111.28**       | 97.59%  |

##### Multi-Threaded

| Matrix Size (M, K, N) | TinyMLP (GFLOPS) | PyTorch (GFLOPS) | Speedup  |
| --------------------- | ---------------- | ---------------- | -------- |
| 18×96×32              | **118.92**       | 52.94            | 224.63%  |
| 24×96×64              | **134.60**       | 0.11             | 1223.64% |
| 24×256×64             | **130.13**       | 114.93           | 113.23%  |
| 48×128×64             | 129.71           | **150.25**       | 86.33%   |
| 192×128×64            | 128.20           | **257.44**       | 49.80%   |
| 192×128×128           | 133.09           | **296.31**       | 44.92%   |
| 480×16×512            | 109.07           | **261.63**       | 41.69%   |
| 192×256×256           | 130.11           | **172.23**       | 75.53%   |
| 384×256×256           | 129.77           | **377.39**       | 51.08%   |
| 480×256×512           | 125.91           | **364.21**       | 34.57%   |
| 1024×1024×1024        | 340.05           | **357.51**       | 95.12%   |
| 1020×1152×1152        | 347.37           | **373.73**       | 92.95%   |
| 1920×2304×2304        | **419.27**       | 334.01           | 125.53%  |
| 2304×2560×2304        | **436.56**       | 313.53           | 139.24%  |

*The Clang version is 21.1.0; the PyTorch version is 2.9.1+cpu; the oneMKL version is 2024.2-Product Build 20240605.*

#### Apple M2 (8GB)

+ **Theoretical GFLOPS**
  + **P-Core**: 3.49 GHz $\times$ 4 floats $\times$ 4 FMA units $\times$ 2 ops (fmul + fadd) $=$ 111.68 GFLOPs
  + **E-Core**: 2.42 GHz $\times$ 4 floats $\times$ 2 FMA units $\times$ 2 ops (fmul + fadd) $=$ 38.72 GFLOPs
+ **Operating System**
  + macOS Sonoma 14.4.1
+ **Comparison**
  + TinyMLP (Apple Clang) vs PyTorch (Apple AMX)

| Matrix Size (M, K, N) | Single-Threaded TinyMLP (GFLOPS) | Multi-Threaded TinyMLP (GFLOPS) | PyTorch (GFLOPS) |
| --------------------- | -------------------------------- | ------------------------------- | ---------------- |
| 18×96×32              | 29.49                            | 29.82                           | 75.80            |
| 24×96×64              | 33.23                            | 33.23                           | 181.48           |
| 24×256×64             | 33.70                            | 33.76                           | 343.12           |
| 48×128×64             | 41.12                            | 33.64                           | 377.55           |
| 192×128×64            | 41.11                            | 41.14                           | 645.28           |
| 192×128×128           | 54.26                            | 54.29                           | 838.86           |
| 480×16×512            | 57.16                            | 57.20                           | 403.30           |
| 192×256×256           | 94.97                            | 94.92                           | 1067.07          |
| 384×256×256           | 95.48                            | 95.59                           | 1067.11          |
| 480×256×512           | 95.06                            | 95.07                           | 1259.34          |
| 1024×1024×1024        | 95.45                            | 360.20                          | 1256.28          |
| 1020×1152×1152        | 94.35                            | 358.88                          | 1393.66          |
| 1920×2304×2304        | 87.53                            | 342.70                          | 1100.45          |
| 2304×2560×2304        | 87.15                            | 331.45                          | 1121.53          |

*The Apple Clang version is 15.0.0; the PyTorch version is 2.8.0; AMX is Apple's undocumented CPU-controlled matmul accelerator.*

## References

+ [Decoupled Weight Decay Regularization](https://openreview.net/pdf?id=Bkg6RiCqY7), ICLR '19
+ [Incorporating Nesterov Momentum into Adam](https://openreview.net/pdf/OM0jvwB8jIp57ZJjtNEZ.pdf), ICLR '16 Workshop
+ [Tile Size Selection Revisited](https://dl.acm.org/doi/10.1145/2541228.2555292), TACO 10 (4), 2013
