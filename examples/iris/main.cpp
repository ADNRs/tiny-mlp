#include "TinyMLP.hpp"
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>

#define DATA_PATH "iris.data"

struct ModelParams {
  using T = float;
  static constexpr size_t FeatDim = 4;
  static constexpr size_t HiddenDim = 32;
  static constexpr size_t LabelDim = 3;
};

template <size_t BatchSize>
using Layers = tinymlp::LayerList<
    tinymlp::Linear<BatchSize, ModelParams::FeatDim, ModelParams::HiddenDim,
                    ModelParams::T>,
    tinymlp::Tanh<BatchSize, ModelParams::HiddenDim, ModelParams::T>,
    tinymlp::Linear<BatchSize, ModelParams::HiddenDim, ModelParams::LabelDim,
                    ModelParams::T>>;

template <size_t BatchSize>
using Model = tinymlp::Sequential<BatchSize, Layers<BatchSize>, ModelParams::T>;

template <size_t BatchSize>
using LossFunc =
    tinymlp::CrossEntropy<BatchSize, ModelParams::LabelDim, ModelParams::T>;

template <size_t BatchSize>
using Trainer = tinymlp::SequentialTrainer<BatchSize, LossFunc<BatchSize>,
                                           Layers<BatchSize>, ModelParams::T>;

static constexpr size_t NumSamples = 150;
static constexpr size_t NumEpochs = 50;
static constexpr size_t BatchSize = 16;

static void
parseIris(const std::string &Path,
          tinymlp::Mat<NumSamples, ModelParams::FeatDim, ModelParams::T> &X,
          tinymlp::Mat<NumSamples, ModelParams::LabelDim, ModelParams::T> &Y) {
  std::ifstream File(Path);
  std::string Line;
  size_t SampleIdx = 0;

  matFill(Y, ModelParams::T(0));

  while (std::getline(File, Line) && SampleIdx < NumSamples) {
    std::stringstream SS(Line);
    std::string Item;
    size_t FeatureIdx = 0;

    while (FeatureIdx < ModelParams::FeatDim) {
      std::getline(SS, Item, ',');
      X(SampleIdx, FeatureIdx) = static_cast<ModelParams::T>(std::stof(Item));
      FeatureIdx++;
    }

    std::getline(SS, Item, ',');

    if (Item == "Iris-setosa")
      Y(SampleIdx, 0) = ModelParams::T(1);
    else if (Item == "Iris-versicolor")
      Y(SampleIdx, 1) = ModelParams::T(1);
    else if (Item == "Iris-virginica")
      Y(SampleIdx, 2) = ModelParams::T(1);

    SampleIdx++;
  }
}

int main() {
  constexpr size_t AllocSize =
      tinymlp::getNearestMultiple(NumSamples, BatchSize);
  auto StorageX = std::make_unique<
      tinymlp::Mat<AllocSize, ModelParams::FeatDim, ModelParams::T>>();
  auto StorageY = std::make_unique<
      tinymlp::Mat<AllocSize, ModelParams::LabelDim, ModelParams::T>>();

  auto &X = (*StorageX).template getView<NumSamples>();
  auto &Y = (*StorageY).template getView<NumSamples>();

  parseIris(DATA_PATH, X, Y);

  std::cout << "[Sample 1]\n";
  for (size_t J = 0; J < ModelParams::FeatDim; J++)
    std::cout << std::fixed << std::setprecision(1) << X(0, J)
              << (J % 7 == 6 ? "\n" : "\t");
  std::cout << "\n[Label 1]\n"
            << (int)Y(0, 0) << "\t" << (int)Y(0, 1) << "\t" << (int)Y(0, 2)
            << "\n";

  tinymlp::shuffle(X, Y);

  auto Dataset = std::make_unique<
      tinymlp::Dataset<NumSamples, ModelParams::FeatDim, ModelParams::LabelDim,
                       BatchSize, ModelParams::T>>(X, Y);

  auto MT = std::make_unique<Trainer<BatchSize>>();

  for (size_t Epoch = 0; Epoch < NumEpochs; Epoch++) {
    size_t Iter = 0;
    ModelParams::T TrainLoss = 0;

    for (const auto &[X, Y] : Dataset->epoch()) {
      ModelParams::T BatchLoss = 0;
      MT->clearGrad();
      MT->propagate(X, Y, BatchLoss);
      MT->update();
      TrainLoss += BatchLoss;
      Iter++;

      std::cout << "\33[2K\r[Epoch " << Epoch + 1 << "/" << NumEpochs
                << "] Batch " << Iter << "/" << Dataset->numBatches()
                << " - Loss: " << std::fixed << std::setprecision(6)
                << (TrainLoss / Iter) << std::flush;
    }
  }

  std::cout << "\n";

  unsigned int Correct = 0;
  auto PredY = std::make_unique<
      tinymlp::Mat<BatchSize, ModelParams::LabelDim, ModelParams::T>>();
  auto PredLabel = std::make_unique<tinymlp::Mat<BatchSize, 1, size_t>>();
  auto TrueLabel = std::make_unique<tinymlp::Mat<BatchSize, 1, size_t>>();

  for (size_t I = 0; I < NumSamples; I += BatchSize) {
    size_t RealBatchSize = std::min(BatchSize, NumSamples - I);
    auto &X = Dataset->X.template getView<BatchSize>(I);
    auto &Y = Dataset->Y.template getView<BatchSize>(I);

    MT->forward(X, *PredY);
    tinymlp::matArgmaxRow(*PredY, *PredLabel);
    tinymlp::matArgmaxRow(Y, *TrueLabel);

    for (size_t J = 0; J < RealBatchSize; J++)
      Correct += (*PredLabel)(J, 0) == (*TrueLabel)(J, 0);
  }

  std::cout << "[Accuracy]\n"
            << std::fixed << std::setprecision(2)
            << (float)Correct / NumSamples * 100.f << "%\n";

  return 0;
}
