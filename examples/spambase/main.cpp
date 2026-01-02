#include "TinyMLP.hpp"
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>

#define DATA_PATH "spambase.data"

struct ModelParams {
  using T = float;
  static constexpr size_t FeatDim = 57;
  static constexpr size_t HiddenDim1 = 32;
  static constexpr size_t HiddenDim2 = 32;
  static constexpr size_t LabelDim = 1;
};

template <size_t BatchSize>
using Layers = tinymlp::LayerList<
    tinymlp::Linear<BatchSize, ModelParams::FeatDim, ModelParams::HiddenDim1,
                    ModelParams::T>,
    tinymlp::ReLU<BatchSize, ModelParams::HiddenDim1, ModelParams::T>,
    tinymlp::Linear<BatchSize, ModelParams::HiddenDim1, ModelParams::HiddenDim2,
                    ModelParams::T>,
    tinymlp::ReLU<BatchSize, ModelParams::HiddenDim2, ModelParams::T>,
    tinymlp::Linear<BatchSize, ModelParams::HiddenDim2, ModelParams::LabelDim,
                    ModelParams::T>>;

template <size_t BatchSize>
using Model = tinymlp::Sequential<BatchSize, Layers<BatchSize>, ModelParams::T>;

template <size_t BatchSize>
using LossFunc = tinymlp::BCEWithLogits<BatchSize, ModelParams::T>;

template <size_t BatchSize>
using Trainer = tinymlp::SequentialTrainer<BatchSize, LossFunc<BatchSize>,
                                           Layers<BatchSize>, ModelParams::T>;

static constexpr size_t NumSamples = 4601;
static constexpr size_t NumEpochs = 350;
static constexpr size_t BatchSize = 4;
static constexpr size_t AccumSteps = 4;

static void parseSpambase(
    const std::string &Path,
    tinymlp::Mat<NumSamples, ModelParams::FeatDim, ModelParams::T> &X,
    tinymlp::Mat<NumSamples, ModelParams::LabelDim, ModelParams::T> &Y) {
  std::ifstream File(Path);
  std::string Line;
  size_t SampleIdx = 0;

  while (std::getline(File, Line)) {
    std::stringstream SS(Line);
    std::string Item;
    size_t FeatureIdx = 0;

    while (FeatureIdx < ModelParams::FeatDim) {
      std::getline(SS, Item, ',');
      X(SampleIdx, FeatureIdx) = static_cast<ModelParams::T>(std::stof(Item));
      FeatureIdx++;
    }

    std::getline(SS, Item, ',');
    Y(SampleIdx, 0) = static_cast<ModelParams::T>(std::stof(Item));

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

  parseSpambase(DATA_PATH, X, Y);
  tinymlp::smoothLabel(Y, 0.1f);

  std::cout << "[Sample 1]\n";
  for (size_t J = 0; J < ModelParams::FeatDim; J++)
    std::cout << X(0, J) << (J % 7 == 6 ? "\n" : "\t");
  std::cout << "\n[Label 1]\n" << Y(0, 0) << "\n";

  tinymlp::shuffle(X, Y);

  auto Dataset = std::make_unique<
      tinymlp::Dataset<NumSamples, ModelParams::FeatDim, ModelParams::LabelDim,
                       BatchSize, ModelParams::T>>(X, Y);

  auto MT = std::make_unique<Trainer<BatchSize>>();

  for (size_t Epoch = 0; Epoch < NumEpochs; Epoch++) {
    size_t Iter = 0;
    ModelParams::T TrainLoss = 0;

    MT->clearGrad();

    for (const auto &[X, Y] : Dataset->epoch()) {
      ModelParams::T BatchLoss = 0;
      MT->propagate(X, Y, BatchLoss);
      TrainLoss += BatchLoss;
      Iter++;

      if (Iter % AccumSteps == 0 || Iter == Dataset->numBatches()) {
        size_t EffectiveAccumSteps =
            (Iter % AccumSteps == 0) ? AccumSteps : (Iter % AccumSteps);
        MT->scaleGrad(1.0f / EffectiveAccumSteps);
        MT->update();
        MT->clearGrad();
      }

      if (Iter % 59 == 1 || Iter == Dataset->numBatches())
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

  for (size_t I = 0; I < NumSamples; I += BatchSize) {
    size_t RealBatchSize = std::min(BatchSize, NumSamples - I);
    auto &X = Dataset->X.template getView<BatchSize>(I);
    auto &Y = Dataset->Y.template getView<BatchSize>(I);

    MT->forward(X, *PredY);

    for (size_t J = 0; J < RealBatchSize; J++)
      Correct +=
          (tinymlp::sigmoid((*PredY)(J, 0)) >= 0.5f) == (Y(J, 0) >= 0.5f);
  }

  std::cout << "[Accuracy]\n"
            << std::fixed << std::setprecision(2)
            << (float)Correct / NumSamples * 100.f << "%\n";

  return 0;
}
