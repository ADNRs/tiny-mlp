#include "TinyMLP.hpp"
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>

#define DATA_PATH "train.csv"
#define WEIGHT_PATH "weights.bin"

struct ModelParams {
  using T = float;
  static constexpr size_t FeatDim = 81;
  static constexpr size_t HiddenDim1 = 1024;
  static constexpr size_t HiddenDim2 = 1024;
  static constexpr size_t LabelDim = 1;
};

struct NAdamWConfig : public tinymlp::DefaultNAdamWConfig<ModelParams::T> {
  static constexpr ModelParams::T Alpha = 3e-4;
  static constexpr ModelParams::T Lambda = 1e-5;
};

struct DropoutConfig : public tinymlp::DefaultDropoutConfig<ModelParams::T> {
  static constexpr ModelParams::T DropProb = 0.2f;
};

template <size_t BatchSize, bool AllowParallel>
using Layers = tinymlp::LayerList<
    tinymlp::Linear<BatchSize, ModelParams::FeatDim, ModelParams::HiddenDim1,
                    ModelParams::T, AllowParallel>,
    tinymlp::ReLU<BatchSize, ModelParams::HiddenDim1, ModelParams::T>,
    tinymlp::Dropout<BatchSize, ModelParams::HiddenDim1, ModelParams::T,
                     DropoutConfig>,
    tinymlp::Linear<BatchSize, ModelParams::HiddenDim1, ModelParams::HiddenDim2,
                    ModelParams::T, AllowParallel>,
    tinymlp::ReLU<BatchSize, ModelParams::HiddenDim2, ModelParams::T>,
    tinymlp::Dropout<BatchSize, ModelParams::HiddenDim2, ModelParams::T,
                     DropoutConfig>,
    tinymlp::Linear<BatchSize, ModelParams::HiddenDim2, ModelParams::LabelDim,
                    ModelParams::T, AllowParallel>>;

template <size_t BatchSize, bool AllowParallel>
using Model = tinymlp::Sequential<BatchSize, Layers<BatchSize, AllowParallel>,
                                  ModelParams::T>;

template <size_t BatchSize>
using LossFunc = tinymlp::MSE<BatchSize, ModelParams::T>;

template <size_t BatchSize, bool AllowParallel>
using Trainer = tinymlp::SequentialTrainer<BatchSize, LossFunc<BatchSize>,
                                           Layers<BatchSize, AllowParallel>,
                                           ModelParams::T, NAdamWConfig>;

static constexpr size_t NumSamples = 21263;
static constexpr size_t NumEpochs = 200;
static constexpr size_t BatchSize = 1024;
static constexpr size_t TrainSize = NumSamples * 0.8;
static constexpr size_t TestSize = NumSamples - TrainSize;

static void parseSuperconductivity(
    const std::string &Path,
    tinymlp::Mat<NumSamples, ModelParams::FeatDim, ModelParams::T> &X,
    tinymlp::Mat<NumSamples, ModelParams::LabelDim, ModelParams::T> &Y) {
  std::ifstream File(Path);
  std::string Line;
  size_t SampleIdx = 0;

  std::getline(File, Line);

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

template <size_t BatchSize>
static void
train(Trainer<BatchSize, true> &ModelTrainer,
      tinymlp::Dataset<TrainSize, ModelParams::FeatDim, ModelParams::LabelDim,
                       BatchSize, ModelParams::T> &TrainingSet) {
  for (size_t Epoch = 0; Epoch < NumEpochs; Epoch++) {
    ModelParams::T TrainLoss = 0;

    size_t Iter = 0;
    for (const auto &[X, Y] : TrainingSet.epoch()) {
      ModelParams::T BatchLoss = 0;
      ModelTrainer.clearGrad();
      ModelTrainer.propagate(X, Y, BatchLoss);
      ModelTrainer.update();
      TrainLoss += BatchLoss;
      Iter++;

      std::cout << "\33[2K\r[Epoch " << Epoch + 1 << "/" << NumEpochs
                << "] Batch " << Iter << "/" << TrainingSet.numBatches()
                << " - Loss: " << std::fixed << std::setprecision(6)
                << (TrainLoss / Iter) << std::flush;
    }

    tinymlp::shuffle(TrainingSet.X, TrainingSet.Y);
  }

  std::cout << "\n";
}

template <size_t BatchSize, typename ModelT>
static ModelParams::T
evaluate(ModelT &Model,
         tinymlp::Dataset<TestSize, ModelParams::FeatDim, ModelParams::LabelDim,
                          1, ModelParams::T> &TestSet) {
  ModelParams::T RMSE = 0;
  auto PredY = std::make_unique<
      tinymlp::Mat<BatchSize, ModelParams::LabelDim, ModelParams::T>>();

  for (size_t I = 0; I < TestSize; I += BatchSize) {
    size_t RealBatchSize = std::min(BatchSize, TestSize - I);
    auto &X = TestSet.X.template getView<BatchSize>(I);
    auto &Y = TestSet.Y.template getView<BatchSize>(I);

    Model.forward(X, *PredY);

    for (size_t J = 0; J < RealBatchSize; J++) {
      ModelParams::T Diff = (*PredY)(J, 0) - Y(J, 0);
      RMSE += Diff * Diff;
    }
  }

  return std::sqrt(RMSE / TestSize);
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

  parseSuperconductivity(DATA_PATH, X, Y);

  std::cout << "[Sample 1]\n";
  for (size_t J = 0; J < ModelParams::FeatDim; J++)
    std::cout << std::right << std::setw(12) << std::fixed
              << std::setprecision(6) << X(0, J) << (J % 7 == 6 ? "\n" : "\t");
  std::cout << "\n[Label 1]\n" << Y(0, 0) << "\n";

  tinymlp::shuffle(X, Y);

  auto &TrainX = X.template getView<TrainSize>();
  auto &TrainY = Y.template getView<TrainSize>();
  auto &TestX = X.template getView<TestSize>(TrainSize);
  auto &TestY = Y.template getView<TestSize>(TrainSize);

  std::cout << "[Training set size]\n"
            << TrainSize << "\n[Test set size]\n"
            << TestSize << "\n";

  auto TrainingSet = std::make_unique<
      tinymlp::Dataset<TrainSize, ModelParams::FeatDim, ModelParams::LabelDim,
                       BatchSize, ModelParams::T>>(TrainX, TrainY);
  auto TestSet = std::make_unique<
      tinymlp::Dataset<TestSize, ModelParams::FeatDim, ModelParams::LabelDim, 1,
                       ModelParams::T>>(TestX, TestY);

  auto MT = std::make_unique<Trainer<BatchSize, true>>();

  train(*MT, *TrainingSet);

  std::cout << "[Test set RMSE]\n"
            << std::fixed << std::setprecision(6)
            << evaluate<BatchSize, Trainer<BatchSize, true>>(*MT, *TestSet)
            << " K\n";

  std::ofstream Out(WEIGHT_PATH, std::ios::binary);
  MT->save(Out);
  Out.close();

  std::cout << "[Save]\n" << WEIGHT_PATH << "\n";

  auto M = std::make_unique<Model<1, false>>();
  std::ifstream In(WEIGHT_PATH, std::ios::binary);
  M->load(In);
  In.close();

  std::cout << "[Load]\n" << WEIGHT_PATH << "\n";
  std::cout << "[Test set RMSE (from loaded weights)]\n"
            << std::fixed << std::setprecision(6)
            << evaluate<1, decltype(*M)>(*M, *TestSet) << " K\n";

  return 0;
}
