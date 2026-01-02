# Examples

There are three examples to showcase the usage of `TinyMLP.hpp`.

+ `iris/`
  + Multi-class classification (`CrossEntropy`)
  + Inference of multi-class classification model
+ `spambase/`
  + Binary classification (`BCEWithLogits`)
  + Inference of binary classification model
  + Label smoothing
  + Gradient accumulation
+ `superconductivity/`
  + Regression (`MSE`)
  + Overriding parameter settings of `Dropout` and `NAdamW`
  + Splitting training and test sets
  + Saving and loading weights
  + Enabling multi-threaded matrix multiplication of the Linear layer

## Usage

```bash
cd <folder> && \
./download.sh && \ # Download the dataset; only needs to be done once
make && \
./main
```

## Notices

### Data Padding for Inference

Because the model takes input in fixed-size matrices, processing the final partial data can trigger out-of-bounds memory access if `NumSamples` is not a multiple of `BatchSize`. In the example below, the model needs two batches to cover the inference of data. To prevent memory errors, we allocate a matrix rounded up to the nearest batch multiple, then create a logical view of the actual data size. This method avoids the performance overhead of copying partial batches into temporary storage. Note that this applies only to inference; `Dataset` handles this issue via the drop-last approach of mini-batches.

```C++
constexpr size_t NumSamples = 100000;
constexpr size_t BatchSize = 99999;
constexpr size_t AllocSize =
    tinymlp::getNearestMultiple(NumSamples, BatchSize);

// std::unique_ptr<tinymlp::Mat<199998, FeatDim, T>>
auto StorageX = std::make_unique<
    tinymlp::Mat<AllocSize, FeatDim, T>>();

// tinymlp::Mat<100000, FeatDim, T> &
auto &X = (*StorageX).template getView<NumSamples>();
```

### Using `BCEWithLogits`

Do not use `Sigmoid` as the final layer during training. `BCEWithLogits` combines the sigmoid operation and the binary cross-entropy loss to improve numerical precision. Consequently, the model outputs raw logits rather than probabilities. For inference, you have two options.

1. Run the model as-is and apply `tinymlp::sigmoid` to the output manually.
2. Define a separate model structure for inference that appends a `tinymlp::Sigmoid` layer.

Note that this behaves exactly like `torch.nn.BCEWithLogitsLoss`.
