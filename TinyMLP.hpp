#ifndef TINYMLP_HPP
#define TINYMLP_HPP

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <fstream>
#include <iterator>
#include <limits>
#include <new>
#include <random>
#include <thread>
#include <tuple>
#include <type_traits>
#include <utility>

#define TINYMLP_ATTRS __attribute__((always_inline)) static inline

namespace tinymlp {

static constexpr size_t Align = 64;
static constexpr size_t RegTM = 4;
static constexpr size_t RegTN = 16;

struct L1Info {
  static constexpr size_t Size = 32 * 1024;
  static constexpr size_t Assoc = 8;
  static constexpr size_t LineSize = 64;
};

struct L2Info {
  static constexpr size_t Size = 256 * 1024;
  static constexpr size_t Assoc = 8;
  static constexpr size_t LineSize = 64;
};

template <typename T>
using Arithmetic = std::enable_if_t<std::is_arithmetic_v<T>>;

template <typename T>
using FloatingPoint = std::enable_if_t<std::is_floating_point_v<T>>;

static constexpr size_t getNearestMultiple(size_t Val, size_t Base) {
  return (Val + Base - 1) / Base * Base;
}

template <size_t M, size_t N, typename T, typename = Arithmetic<T>,
          typename = std::enable_if_t<Align >= sizeof(T)>>
struct Mat {
  static constexpr size_t AllocM = getNearestMultiple(M, RegTM);
  static constexpr size_t AllocN =
      getNearestMultiple(N, std::gcd(RegTN, Align / sizeof(T))) +
      Align / sizeof(T) * 3;

private:
  alignas(Align) T Data[AllocM * AllocN]{};

public:
  T &operator()(size_t Row, size_t Col) { return Data[Row * AllocN + Col]; }

  T operator()(size_t Row, size_t Col) const {
    return Data[Row * AllocN + Col];
  }

  T *getAddr(size_t Row, size_t Col) { return &Data[Row * AllocN + Col]; }

  const T *getAddr(size_t Row, size_t Col) const {
    return &Data[Row * AllocN + Col];
  }

  template <size_t NewM> Mat<NewM, N, T> &getView(size_t Row = 0) {
    T *BaseAddr = getAddr(Row, 0);
    void *AlignedPtr = __builtin_assume_aligned(BaseAddr, Align);
    return *std::launder(reinterpret_cast<Mat<NewM, N, T> *>(AlignedPtr));
  }
};

template <size_t M, size_t N, typename T, typename U = T,
          typename = Arithmetic<T>, typename = Arithmetic<U>>
TINYMLP_ATTRS void matCopy(const Mat<M, N, T> &A, Mat<M, N, U> &B) {
  for (size_t I = 0; I < M; ++I)
    for (size_t J = 0; J < N; ++J)
      B(I, J) = A(I, J);
}

template <size_t M, size_t N, typename T, typename U = T,
          typename = Arithmetic<T>, typename = Arithmetic<U>>
TINYMLP_ATTRS void matTrans(const Mat<M, N, T> &A, Mat<N, M, U> &AT) {
  for (size_t I = 0; I < M; ++I)
    for (size_t J = 0; J < N; ++J)
      AT(J, I) = A(I, J);
}

template <size_t M, size_t N, typename T, typename U = T,
          typename = Arithmetic<T>, typename = Arithmetic<U>>
TINYMLP_ATTRS void matFill(Mat<M, N, T> &A, U Val) {
  for (size_t I = 0; I < M; ++I)
    for (size_t J = 0; J < N; ++J)
      A(I, J) = Val;
}

struct TileSize {
  size_t TM;
  size_t TK;
  size_t TN;
};

template <size_t N, size_t CLS, size_t NumSets, typename Config>
TINYMLP_ATTRS constexpr size_t getTileHeight(size_t W) {
  size_t CacheEmu[Config::Size / Config::LineSize]{};

  for (size_t R = 0; R < Config::Size / Config::LineSize; ++R) {
    size_t NSet = (R * N / CLS) % NumSets;

    for (size_t C = 0; C <= W / CLS; ++C) {
      if (CacheEmu[NSet + C] == Config::Assoc - 1)
        return R;
      ++CacheEmu[NSet + C];
    }
  }

  return 1;
}

template <size_t N, typename T, typename = Arithmetic<T>>
TINYMLP_ATTRS constexpr TileSize runTSS() {
  TileSize TS = {1, 1, 1};

  double MinCost = std::numeric_limits<double>::max();

  constexpr size_t L2CLS = L2Info::LineSize / sizeof(T);
  constexpr size_t L2NumSets =
      L2Info::Size / (L2Info::LineSize * L2Info::Assoc);

  for (size_t W = 1; W <= L2NumSets; ++W) {
    size_t CurrTN = W * L2CLS;
    if (CurrTN > N)
      break;

    size_t CurrTM = getTileHeight<N, L2CLS, L2NumSets, L2Info>(CurrTN);
    if (CurrTM > L2CLS)
      CurrTM = CurrTM / L2CLS * L2CLS;

    double Cost = (1.0 / CurrTM) + (1.0 / CurrTN);
    if (Cost < MinCost) {
      MinCost = Cost;
      TS.TM = CurrTM;
      TS.TN = CurrTN;
    }
  }

  constexpr size_t L1CLS = L1Info::LineSize / sizeof(T);
  constexpr size_t L1NumSets =
      L1Info::Size / (L1Info::LineSize * L1Info::Assoc);

  size_t BestTK = getTileHeight<N, L1CLS, L1NumSets, L1Info>(TS.TN);
  if (BestTK > L1CLS)
    BestTK = BestTK / L1CLS * L1CLS;

  TS.TM = (TS.TM < RegTM) ? RegTM : (TS.TM / RegTM) * RegTM;
  TS.TK = std::min(BestTK, RegTN);
  TS.TN = (TS.TN < RegTN) ? RegTN : (TS.TN / RegTN) * RegTN;
  return TS;
}

template <size_t M, size_t K, size_t N, size_t TM, size_t TK, size_t TN,
          typename T, typename U = T, typename V = T, typename = Arithmetic<T>,
          typename = Arithmetic<U>, typename = Arithmetic<V>>
TINYMLP_ATTRS void matMulPartial(const Mat<M, K, T> &A, const Mat<K, N, U> &B,
                                 Mat<M, N, V> &C, size_t StartM,
                                 size_t StartN) {
  const size_t EndOM = std::min(StartM + TM, M);
  const size_t EndON = std::min(StartN + TN, N);

  for (size_t StartK = 0; StartK < K; StartK += TK) {
    const size_t EndK = std::min(StartK + TK, K);

    for (size_t InnerM = StartM; InnerM < EndOM; InnerM += RegTM) {
      for (size_t InnerN = StartN; InnerN < EndON; InnerN += RegTN) {
        Mat<RegTM, RegTN, V> RegC;

        for (size_t X = 0; X < RegTM; ++X)
          for (size_t Y = 0; Y < RegTN; ++Y)
            RegC(X, Y) = C(InnerM + X, InnerN + Y);

        for (size_t Z = StartK; Z < EndK; ++Z) {
          Mat<RegTM, 1, T> RegA;
          Mat<1, RegTN, U> RegB;

          for (size_t X = 0; X < RegTM; ++X)
            RegA(X, 0) = A(InnerM + X, Z);

          for (size_t Y = 0; Y < RegTN; ++Y)
            RegB(0, Y) = B(Z, InnerN + Y);

          for (size_t X = 0; X < RegTM; ++X)
            for (size_t Y = 0; Y < RegTN; ++Y)
              RegC(X, Y) += RegA(X, 0) * RegB(0, Y);
        }

        for (size_t X = 0; X < RegTM; ++X)
          for (size_t Y = 0; Y < RegTN; ++Y)
            C(InnerM + X, InnerN + Y) = RegC(X, Y);
      }
    }
  }
}

template <size_t M, size_t K, size_t N, typename T, typename U = T,
          typename V = T, bool AllowParallel = false, typename = Arithmetic<T>,
          typename = Arithmetic<U>, typename = Arithmetic<V>>
TINYMLP_ATTRS void matMul(const Mat<M, K, T> &A, const Mat<K, N, U> &B,
                          Mat<M, N, V> &C) {
  constexpr TileSize TS = runTSS<Mat<M, N, V>::AllocN, T>();
  constexpr size_t TM = (TS.TM < RegTM) ? RegTM : (TS.TM / RegTM) * RegTM;
  constexpr size_t TN = (TS.TN < RegTN) ? RegTN : (TS.TN / RegTN) * RegTN;
  constexpr size_t TK = TS.TK;

  if constexpr (AllowParallel &&
                std::min(TM, M) * std::min(TN, N) * K >= 256 * 256 * 256) {
    if (std::thread::hardware_concurrency() > 1) {
      std::vector<std::thread> Threads;
      const size_t NumWorkers = std::thread::hardware_concurrency() - 1;
      Threads.reserve(NumWorkers);

      std::vector<std::pair<size_t, size_t>> Tasks;

      for (size_t StartM = 0; StartM < M; StartM += TM)
        for (size_t StartN = 0; StartN < N; StartN += TN)
          Tasks.emplace_back(StartM, StartN);

      const size_t NumTasksPerWorker =
          Tasks.size() / std::thread::hardware_concurrency();
      size_t NumRemainedTasks =
          Tasks.size() % std::thread::hardware_concurrency();

      size_t StartI = 0;

      for (size_t WorkerI = 0; WorkerI < NumWorkers; ++WorkerI) {
        size_t NumTasksThisWorker =
            NumTasksPerWorker + (NumRemainedTasks > 0 ? 1 : 0);
        if (NumRemainedTasks > 0)
          --NumRemainedTasks;
        size_t EndI = StartI + NumTasksThisWorker;

        Threads.emplace_back([=, &A, &B, &C, &Tasks]() {
          for (size_t I = StartI; I < EndI; ++I) {
            auto [StartM, StartN] = Tasks[I];
            matMulPartial<M, K, N, TM, TK, TN, T, U, V>(A, B, C, StartM,
                                                        StartN);
          }
        });

        StartI = EndI;
      }

      for (size_t I = StartI; I < Tasks.size(); ++I) {
        auto [StartM, StartN] = Tasks[I];
        matMulPartial<M, K, N, TM, TK, TN, T, U, V>(A, B, C, StartM, StartN);
      }

      for (auto &Th : Threads)
        Th.join();

      return;
    }
  }

  for (size_t StartM = 0; StartM < M; StartM += TM)
    for (size_t StartN = 0; StartN < N; StartN += TN)
      matMulPartial<M, K, N, TM, TK, TN, T, U, V>(A, B, C, StartM, StartN);
}

template <size_t M, size_t N, typename T, typename U = T, typename V = T,
          typename = Arithmetic<T>, typename = Arithmetic<U>,
          typename = Arithmetic<V>>
TINYMLP_ATTRS void matElemWise(const Mat<M, N, T> &A, const Mat<M, N, U> &B,
                               Mat<M, N, V> &C, V (*Func)(T, U)) {
  for (size_t I = 0; I < M; ++I)
    for (size_t J = 0; J < N; ++J)
      C(I, J) = Func(A(I, J), B(I, J));
}

template <size_t M, size_t N, typename T, typename U = T, typename V = T,
          typename = Arithmetic<T>, typename = Arithmetic<U>,
          typename = Arithmetic<V>>
TINYMLP_ATTRS void matAdd(const Mat<M, N, T> &A, const Mat<M, N, U> &B,
                          Mat<M, N, V> &C) {
  matElemWise(A, B, C, +[](T Val1, U Val2) { return (V)(Val1 + Val2); });
}

template <size_t M, size_t N, typename T, typename U = T, typename V = T,
          typename = Arithmetic<T>, typename = Arithmetic<U>,
          typename = Arithmetic<V>>
TINYMLP_ATTRS void matSub(const Mat<M, N, T> &A, const Mat<M, N, U> &B,
                          Mat<M, N, V> &C) {
  matElemWise(A, B, C, +[](T Val1, U Val2) { return (V)(Val1 - Val2); });
}

template <size_t M, size_t N, typename T, typename U = T, typename V = T,
          typename = Arithmetic<T>, typename = Arithmetic<U>,
          typename = Arithmetic<V>>
TINYMLP_ATTRS void matHadamard(const Mat<M, N, T> &A, const Mat<M, N, U> &B,
                               Mat<M, N, V> &C) {
  matElemWise(A, B, C, +[](T Val1, U Val2) { return (V)(Val1 * Val2); });
}

template <size_t M, size_t N, typename T, typename U = T, typename V = T,
          typename = Arithmetic<T>, typename = Arithmetic<U>,
          typename = Arithmetic<V>>
TINYMLP_ATTRS void matDiv(const Mat<M, N, T> &A, const Mat<M, N, U> &B,
                          Mat<M, N, V> &C) {
  matElemWise(A, B, C, +[](T Val1, U Val2) { return (V)(Val1 / Val2); });
}

template <size_t M, size_t N, typename T, typename U = T, typename V = T,
          typename = Arithmetic<T>, typename = Arithmetic<U>,
          typename = Arithmetic<V>>
TINYMLP_ATTRS void matElemWiseRowVec(const Mat<M, N, T> &A,
                                     const Mat<1, N, U> &B, Mat<M, N, V> &C,
                                     V (*Func)(T, U)) {
  for (size_t I = 0; I < M; ++I)
    for (size_t J = 0; J < N; ++J)
      C(I, J) = Func(A(I, J), B(0, J));
}

template <size_t M, size_t N, typename T, typename U = T, typename V = T,
          typename = Arithmetic<T>, typename = Arithmetic<U>,
          typename = Arithmetic<V>>
TINYMLP_ATTRS void matAddRowVec(const Mat<M, N, T> &A, const Mat<1, N, U> &B,
                                Mat<M, N, V> &C) {
  matElemWiseRowVec(A, B, C, +[](T Val1, U Val2) { return (V)(Val1 + Val2); });
}

template <size_t M, size_t N, typename T, typename U = T, typename V = T,
          typename = Arithmetic<T>, typename = Arithmetic<U>,
          typename = Arithmetic<V>>
TINYMLP_ATTRS void matSubRowVec(const Mat<M, N, T> &A, const Mat<1, N, U> &B,
                                Mat<M, N, V> &C) {
  matElemWiseRowVec(A, B, C, +[](T Val1, U Val2) { return (V)(Val1 - Val2); });
}

template <size_t M, size_t N, typename T, typename U = T, typename V = T,
          typename = Arithmetic<T>, typename = Arithmetic<U>,
          typename = Arithmetic<V>>
TINYMLP_ATTRS void matHadamardRowVec(const Mat<M, N, T> &A,
                                     const Mat<1, N, U> &B, Mat<M, N, V> &C) {
  matElemWiseRowVec(A, B, C, +[](T Val1, U Val2) { return (V)(Val1 * Val2); });
}

template <size_t M, size_t N, typename T, typename U = T, typename V = T,
          typename = Arithmetic<T>, typename = Arithmetic<U>,
          typename = Arithmetic<V>>
TINYMLP_ATTRS void matDivRowVec(const Mat<M, N, T> &A, const Mat<1, N, U> &B,
                                Mat<M, N, V> &C) {
  matElemWiseRowVec(A, B, C, +[](T Val1, U Val2) { return (V)(Val1 / Val2); });
}

template <size_t M, size_t N, typename T, typename U = T, typename V = T,
          typename = Arithmetic<T>, typename = Arithmetic<U>,
          typename = Arithmetic<V>>
TINYMLP_ATTRS void matElemWiseColVec(const Mat<M, N, T> &A,
                                     const Mat<M, 1, U> &B, Mat<M, N, V> &C,
                                     V (*Func)(T, U)) {
  for (size_t I = 0; I < M; ++I) {
    U Val = B(I, 0);
    for (size_t J = 0; J < N; ++J)
      C(I, J) = Func(A(I, J), Val);
  }
}

template <size_t M, size_t N, typename T, typename U = T, typename V = T,
          typename = Arithmetic<T>, typename = Arithmetic<U>,
          typename = Arithmetic<V>>
TINYMLP_ATTRS void matAddColVec(const Mat<M, N, T> &A, const Mat<M, 1, U> &B,
                                Mat<M, N, V> &C) {
  matElemWiseColVec(A, B, C, +[](T Val1, U Val2) { return (V)(Val1 + Val2); });
}

template <size_t M, size_t N, typename T, typename U = T, typename V = T,
          typename = Arithmetic<T>, typename = Arithmetic<U>,
          typename = Arithmetic<V>>
TINYMLP_ATTRS void matSubColVec(const Mat<M, N, T> &A, const Mat<M, 1, U> &B,
                                Mat<M, N, V> &C) {
  matElemWiseColVec(A, B, C, +[](T Val1, U Val2) { return (V)(Val1 - Val2); });
}

template <size_t M, size_t N, typename T, typename U = T, typename V = T,
          typename = Arithmetic<T>, typename = Arithmetic<U>,
          typename = Arithmetic<V>>
TINYMLP_ATTRS void matHadamardColVec(const Mat<M, N, T> &A,
                                     const Mat<M, 1, U> &B, Mat<M, N, V> &C) {
  matElemWiseColVec(A, B, C, +[](T Val1, U Val2) { return (V)(Val1 * Val2); });
}

template <size_t M, size_t N, typename T, typename U = T, typename V = T,
          typename = Arithmetic<T>, typename = Arithmetic<U>,
          typename = Arithmetic<V>>
TINYMLP_ATTRS void matDivColVec(const Mat<M, N, T> &A, const Mat<M, 1, U> &B,
                                Mat<M, N, V> &C) {
  matElemWiseColVec(A, B, C, +[](T Val1, U Val2) { return (V)(Val1 / Val2); });
}

template <size_t M, size_t N, typename T, typename U = T, typename V = T,
          typename = Arithmetic<T>, typename = Arithmetic<U>,
          typename = Arithmetic<V>>
TINYMLP_ATTRS void matElemWiseScalar(const Mat<M, N, T> &A, U B,
                                     Mat<M, N, V> &C, V (*Func)(T, U)) {
  for (size_t I = 0; I < M; ++I)
    for (size_t J = 0; J < N; ++J)
      C(I, J) = Func(A(I, J), B);
}

template <size_t M, size_t N, typename T, typename U = T, typename V = T,
          typename = Arithmetic<T>, typename = Arithmetic<U>,
          typename = Arithmetic<V>>
TINYMLP_ATTRS void matAddScalar(const Mat<M, N, T> &A, U B, Mat<M, N, V> &C) {
  matElemWiseScalar(A, B, C, +[](T Val1, U Val2) { return (V)(Val1 + Val2); });
}

template <size_t M, size_t N, typename T, typename U = T, typename V = T,
          typename = Arithmetic<T>, typename = Arithmetic<U>,
          typename = Arithmetic<V>>
TINYMLP_ATTRS void matSubScalar(const Mat<M, N, T> &A, U B, Mat<M, N, V> &C) {
  matElemWiseScalar(A, B, C, +[](T Val1, U Val2) { return (V)(Val1 - Val2); });
}

template <size_t M, size_t N, typename T, typename U = T, typename V = T,
          typename = Arithmetic<T>, typename = Arithmetic<U>,
          typename = Arithmetic<V>>
TINYMLP_ATTRS void matHadamardScalar(const Mat<M, N, T> &A, U B,
                                     Mat<M, N, V> &C) {
  matElemWiseScalar(A, B, C, +[](T Val1, U Val2) { return (V)(Val1 * Val2); });
}

template <size_t M, size_t N, typename T, typename U = T, typename V = T,
          typename = Arithmetic<T>, typename = Arithmetic<U>,
          typename = Arithmetic<V>>
TINYMLP_ATTRS void matDivScalar(const Mat<M, N, T> &A, U B, Mat<M, N, V> &C) {
  matElemWiseScalar(A, B, C, +[](T Val1, U Val2) { return (V)(Val1 / Val2); });
}

template <size_t M, size_t N, typename T, typename = Arithmetic<T>>
TINYMLP_ATTRS void matReduceRow(const Mat<M, N, T> &In, Mat<M, 1, T> &Out,
                                T (*Func)(T Val1, T Val2), T InitVal) {
  for (size_t I = 0; I < M; ++I) {
    T Acc = InitVal;
    for (size_t J = 0; J < N; ++J)
      Acc = Func(Acc, In(I, J));
    Out(I, 0) = Acc;
  }
}

template <size_t M, size_t N, typename T, typename = Arithmetic<T>>
TINYMLP_ATTRS void matSumRow(const Mat<M, N, T> &In, Mat<M, 1, T> &Out) {
  matReduceRow(In, Out, +[](T Val1, T Val2) { return Val1 + Val2; }, (T)0);
}

template <size_t M, size_t N, typename T, typename = Arithmetic<T>>
TINYMLP_ATTRS void matMaxRow(const Mat<M, N, T> &In, Mat<M, 1, T> &Out) {
  matReduceRow(
      In, Out, +[](T Val1, T Val2) { return Val1 > Val2 ? Val1 : Val2; },
      std::numeric_limits<T>::lowest());
}

template <size_t M, size_t N, typename T, typename = Arithmetic<T>>
TINYMLP_ATTRS void matReduceCol(const Mat<M, N, T> &In, Mat<1, N, T> &Out,
                                T (*Func)(T Val1, T Val2), T InitVal) {
  matFill(Out, InitVal);
  for (size_t I = 0; I < M; ++I)
    for (size_t J = 0; J < N; ++J)
      Out(0, J) = Func(Out(0, J), In(I, J));
}

template <size_t M, size_t N, typename T, typename = Arithmetic<T>>
TINYMLP_ATTRS void matSumCol(const Mat<M, N, T> &In, Mat<1, N, T> &Out) {
  matReduceCol(In, Out, +[](T Val1, T Val2) { return Val1 + Val2; }, (T)0);
}

template <size_t M, size_t N, typename T, typename = Arithmetic<T>>
TINYMLP_ATTRS void matReduce(const Mat<M, N, T> &In, T &Out,
                             T (*Func)(T Val1, T Val2), T InitVal) {
  T Acc = InitVal;
  for (size_t I = 0; I < M; ++I)
    for (size_t J = 0; J < N; ++J)
      Acc = Func(Acc, In(I, J));
  Out = Acc;
}

template <size_t M, size_t N, typename T, typename = Arithmetic<T>>
TINYMLP_ATTRS void matSum(const Mat<M, N, T> &In, T &Out) {
  matReduce(In, Out, +[](T Val1, T Val2) { return Val1 + Val2; }, (T)0);
}

template <size_t M, size_t N, typename T, typename = Arithmetic<T>>
TINYMLP_ATTRS void matApply(const Mat<M, N, T> &In, Mat<M, N, T> &Out,
                            T (*Func)(T)) {
  for (size_t I = 0; I < M; ++I)
    for (size_t J = 0; J < N; ++J)
      Out(I, J) = Func(In(I, J));
}

template <size_t M, size_t N, typename T, typename = Arithmetic<T>>
TINYMLP_ATTRS void matArgcmpRow(const Mat<M, N, T> &In, Mat<M, 1, size_t> &Out,
                                bool (*Func)(T Val1, T Val2)) {
  for (size_t I = 0; I < M; ++I) {
    T BestVal = In(I, 0);
    size_t BestIdx = 0;
    for (size_t J = 1; J < N; ++J) {
      T Val = In(I, J);
      if (Func(Val, BestVal)) {
        BestVal = Val;
        BestIdx = J;
      }
    }
    Out(I, 0) = BestIdx;
  }
}

template <size_t M, size_t N, typename T, typename = Arithmetic<T>>
TINYMLP_ATTRS void matArgmaxRow(const Mat<M, N, T> &In,
                                Mat<M, 1, size_t> &Out) {
  matArgcmpRow(In, Out, +[](T Val1, T Val2) { return Val1 > Val2; });
}

template <size_t M, size_t N, typename T, typename = Arithmetic<T>>
TINYMLP_ATTRS void matArgminRow(const Mat<M, N, T> &In,
                                Mat<M, 1, size_t> &Out) {
  matArgcmpRow(In, Out, +[](T Val1, T Val2) { return Val1 < Val2; });
}

template <size_t M, size_t N, typename T, typename = Arithmetic<T>>
TINYMLP_ATTRS void writeMat(std::ofstream &Out, const Mat<M, N, T> &A) {
  for (size_t I = 0; I < M; ++I)
    for (size_t J = 0; J < N; ++J)
      Out.write(reinterpret_cast<const char *>(A.getAddr(I, J)), sizeof(T));
}

template <size_t M, size_t N, typename T, typename = Arithmetic<T>>
TINYMLP_ATTRS void readMat(std::ifstream &In, Mat<M, N, T> &A) {
  for (size_t I = 0; I < M; ++I)
    for (size_t J = 0; J < N; ++J)
      In.read(reinterpret_cast<char *>(A.getAddr(I, J)), sizeof(T));
}

template <template <typename...> class Op, typename = void, typename... Args>
struct IsDetected : std::false_type {};

template <template <typename...> class Op, typename... Args>
struct IsDetected<Op, std::void_t<Op<Args...>>, Args...> : std::true_type {};

template <typename T, typename BackpropAux, typename Config>
using UpdateFunc = decltype(std::declval<T>().template update<Config>(
    std::declval<BackpropAux &>()));

template <typename T, typename BackpropAux>
using clearGradFunc =
    decltype(std::declval<T>().clearGrad(std::declval<BackpropAux &>()));

template <typename T, typename ValT, typename BackpropAux>
using ScaleGradFunc = decltype(std::declval<T>().scaleGrad(
    std::declval<ValT>(), std::declval<BackpropAux &>()));

template <typename T>
using SaveFunc =
    decltype(std::declval<T>().save(std::declval<std::ofstream &>()));

template <typename T>
using LoadFunc =
    decltype(std::declval<T>().load(std::declval<std::ifstream &>()));

template <typename T> using InitFunc = decltype(std::declval<T>().init());

template <typename Derived> struct Layer {
  Derived &impl() { return *static_cast<Derived *>(this); }

  template <typename InMat, typename OutMat>
  void forward(const InMat &X, OutMat &Y) {
    impl().forward(X, Y);
  }

  template <typename InMat, typename OutMat, typename BackpropAux>
  void forward(const InMat &X, OutMat &Y, BackpropAux &Aux) {
    impl().forward(X, Y, Aux);
  }

  template <typename InMat, typename OutMat, typename BackpropAux>
  void backward(const InMat &DY, OutMat &DX, BackpropAux &Aux) {
    impl().backward(DY, DX, Aux);
  }

  template <typename Config, typename BackpropAux>
  void update(BackpropAux &Aux) {
    if constexpr (IsDetected<UpdateFunc, Derived, BackpropAux, Config>::value)
      impl().template update<Config>(Aux);
  }

  template <typename BackpropAux> void clearGrad(BackpropAux &Aux) {
    if constexpr (IsDetected<clearGradFunc, Derived, BackpropAux>::value)
      impl().template clearGrad<void>(Aux);
  }

  template <typename ValT, typename BackpropAux>
  void scaleGrad(ValT Factor, BackpropAux &Aux) {
    if constexpr (IsDetected<ScaleGradFunc, Derived, ValT, BackpropAux>::value)
      impl().scaleGrad(Factor, Aux);
  }

  void save(std::ofstream &Out) {
    if constexpr (IsDetected<SaveFunc, Derived>::value)
      impl().save(Out);
  }

  void load(std::ifstream &In) {
    if constexpr (IsDetected<LoadFunc, Derived>::value)
      impl().load(In);
  }

  void init() {
    if constexpr (IsDetected<InitFunc, Derived>::value)
      impl().init();
  }
};

template <size_t M, size_t N, typename T, typename = FloatingPoint<T>>
struct NAdamWState {
  Mat<M, N, T> M1;
  Mat<M, N, T> M2;
  size_t TimeStep = 0;
  T MuProd = 1;

  NAdamWState() {
    matFill(M1, (T)0);
    matFill(M2, (T)0);
  }
};

template <typename T, typename = Arithmetic<T>> struct DefaultNAdamWConfig {
  static constexpr T Alpha = 1e-3;
  static constexpr T Beta1 = 0.9;
  static constexpr T Beta2 = 0.999;
  static constexpr T Eps = 1e-8;
  static constexpr T Lambda = 1e-2;
  static constexpr T Psi = 4e-3;
};

template <size_t M, size_t N, typename T,
          typename Config = DefaultNAdamWConfig<T>, typename = FloatingPoint<T>>
TINYMLP_ATTRS void NAdamW(Mat<M, N, T> &Val, Mat<M, N, T> &Grad,
                          NAdamWState<M, N, T> &State) {
  State.TimeStep++;
  T CurrMu =
      Config::Beta1 * (1 - std::pow(0.96, State.TimeStep * Config::Psi) * 0.5);
  T NextMu = Config::Beta1 *
             (1 - std::pow(0.96, (State.TimeStep + 1) * Config::Psi) * 0.5);
  State.MuProd *= CurrMu;

  for (size_t I = 0; I < M; ++I) {
    for (size_t J = 0; J < N; ++J) {
      T G = Grad(I, J);
      State.M1(I, J) = CurrMu * State.M1(I, J) + (1 - CurrMu) * G;
      State.M2(I, J) =
          Config::Beta2 * State.M2(I, J) + (1 - Config::Beta2) * G * G;
      T M1Hat = NextMu * State.M1(I, J) / (1 - State.MuProd * NextMu) +
                (1 - CurrMu) * G / (1 - State.MuProd);
      T M2Hat = State.M2(I, J) / (1 - std::pow(Config::Beta2, State.TimeStep));
      Val(I, J) -= Config::Alpha * Config::Lambda * Val(I, J);
      Val(I, J) -= Config::Alpha * M1Hat / (std::sqrt(M2Hat) + Config::Eps);
    }
  }
}

template <size_t BatchSize, size_t InDim, size_t OutDim, typename T,
          bool AllowParallel = false, typename = FloatingPoint<T>>
struct Linear : public Layer<Linear<BatchSize, InDim, OutDim, T>> {
  struct BackpropAux {
    Mat<InDim, BatchSize, T> XT;
    Mat<OutDim, InDim, T> WT;
    Mat<InDim, OutDim, T> DW;
    Mat<1, OutDim, T> DB;
    NAdamWState<InDim, OutDim, T> StateW;
    NAdamWState<1, OutDim, T> StateB;
  };

  static constexpr size_t InSize = InDim;
  static constexpr size_t OutSize = OutDim;

  Mat<InDim, OutDim, T> W;
  Mat<1, OutDim, T> B;

  void forward(const Mat<BatchSize, InDim, T> &X,
               Mat<BatchSize, OutDim, T> &Y) {
    matFill(Y, (T)0);
    matMul<BatchSize, InDim, OutDim, T, T, T, AllowParallel>(X, W, Y);
    matAddRowVec(Y, B, Y);
  }

  void forward(const Mat<BatchSize, InDim, T> &X, Mat<BatchSize, OutDim, T> &Y,
               BackpropAux &Aux) {
    matTrans(X, Aux.XT);
    matTrans(W, Aux.WT);
    forward(X, Y);
  }

  void backward(const Mat<BatchSize, OutDim, T> &DY,
                Mat<BatchSize, InDim, T> &DX, BackpropAux &Aux) {
    matMul<InDim, BatchSize, OutDim, T, T, T, AllowParallel>(Aux.XT, DY,
                                                             Aux.DW);
    matSumCol(DY, Aux.DB);
    matFill(DX, (T)0);
    matMul<BatchSize, OutDim, InDim, T, T, T, AllowParallel>(DY, Aux.WT, DX);
  }

  template <typename Config> void update(BackpropAux &Aux) {
    NAdamW<InDim, OutDim, T, Config>(W, Aux.DW, Aux.StateW);
    NAdamW<1, OutDim, T, Config>(B, Aux.DB, Aux.StateB);
  }

  void clearGrad(BackpropAux &Aux) {
    matFill(Aux.DW, (T)0);
    matFill(Aux.DB, (T)0);
  }

  void scaleGrad(T Factor, BackpropAux &Aux) {
    matHadamardScalar(Aux.DW, Factor, Aux.DW);
    matHadamardScalar(Aux.DB, Factor, Aux.DB);
  }

  void save(std::ofstream &Out) {
    writeMat(Out, W);
    writeMat(Out, B);
  }

  void load(std::ifstream &In) {
    readMat(In, W);
    readMat(In, B);
  }

  void init() {
    std::mt19937_64 Gen(42);
    std::normal_distribution<T> Dist(0.0f, std::sqrt(2.0f / (InDim + OutDim)));
    const double Bound = 1 / std::sqrt(InDim);
    std::uniform_real_distribution<T> UniDist(-Bound, Bound);

    for (size_t I = 0; I < InDim; ++I)
      for (size_t J = 0; J < OutDim; ++J)
        W(I, J) = Dist(Gen);

    for (size_t J = 0; J < OutDim; ++J)
      B(0, J) = UniDist(Gen);
  }
};

template <size_t BatchSize, size_t Dim, typename T, typename = FloatingPoint<T>>
struct ReLU : public Layer<ReLU<BatchSize, Dim, T>> {
  struct BackpropAux {
    Mat<BatchSize, Dim, bool> Mask;
  };

  static constexpr size_t InSize = Dim;
  static constexpr size_t OutSize = Dim;

  void forward(const Mat<BatchSize, Dim, T> &X, Mat<BatchSize, Dim, T> &Y) {
    matApply(X, Y, +[](T Val) { return std::max(Val, (T)0); });
  }

  void forward(const Mat<BatchSize, Dim, T> &X, Mat<BatchSize, Dim, T> &Y,
               BackpropAux &Aux) {
    for (size_t I = 0; I < BatchSize; ++I) {
      for (size_t J = 0; J < Dim; ++J) {
        bool Mask = X(I, J) > 0;
        Aux.Mask(I, J) = Mask;
        Y(I, J) = Mask ? X(I, J) : (T)0;
      }
    }
  }

  void backward(const Mat<BatchSize, Dim, T> &DY, Mat<BatchSize, Dim, T> &DX,
                BackpropAux &Aux) {
    matHadamard(DY, Aux.Mask, DX);
  }
};

template <typename T, typename = FloatingPoint<T>>
TINYMLP_ATTRS T sigmoid(T Val) {
  return (T)1 / ((T)1 + std::exp(-Val));
}

template <size_t BatchSize, size_t Dim, typename T, typename = FloatingPoint<T>>
struct Sigmoid : public Layer<Sigmoid<BatchSize, Dim, T>> {
  struct BackpropAux {
    Mat<BatchSize, Dim, T> Y;
  };

  static constexpr size_t InSize = Dim;
  static constexpr size_t OutSize = Dim;

  void forward(const Mat<BatchSize, Dim, T> &X, Mat<BatchSize, Dim, T> &Y) {
    matApply(X, Y, +[](T Val) { return sigmoid(Val); });
  }

  void forward(const Mat<BatchSize, Dim, T> &X, Mat<BatchSize, Dim, T> &Y,
               BackpropAux &Aux) {
    forward(X, Y);
    matCopy(Y, Aux.Y);
  }

  void backward(const Mat<BatchSize, Dim, T> &DY, Mat<BatchSize, Dim, T> &DX,
                BackpropAux &Aux) {
    matApply(Aux.Y, DX, +[](T Val) { return Val * (1 - Val); });
    matHadamard(DX, DY, DX);
  }
};

template <size_t BatchSize, size_t Dim, typename T, typename = FloatingPoint<T>>
struct Tanh : public Layer<Tanh<BatchSize, Dim, T>> {
  struct BackpropAux {
    Mat<BatchSize, Dim, T> Y;
  };

  static constexpr size_t InSize = Dim;
  static constexpr size_t OutSize = Dim;

  void forward(const Mat<BatchSize, Dim, T> &X, Mat<BatchSize, Dim, T> &Y) {
    matApply(X, Y, +[](T Val) { return std::tanh(Val); });
  }

  void forward(const Mat<BatchSize, Dim, T> &X, Mat<BatchSize, Dim, T> &Y,
               BackpropAux &Aux) {
    forward(X, Y);
    matCopy(Y, Aux.Y);
  }

  void backward(const Mat<BatchSize, Dim, T> &DY, Mat<BatchSize, Dim, T> &DX,
                BackpropAux &Aux) {
    matApply(Aux.Y, DX, +[](T Val) { return (T)1 - Val * Val; });
    matHadamard(DX, DY, DX);
  }
};

template <typename T, typename = FloatingPoint<T>> struct DefaultDropoutConfig {
  static constexpr T DropProb = 0.5f;
};

template <size_t BatchSize, size_t Dim, typename T,
          typename Config = DefaultDropoutConfig<T>,
          typename = FloatingPoint<T>>
struct Dropout : public Layer<Dropout<BatchSize, Dim, T, Config>> {
  struct BackpropAux {
    std::mt19937 Gen{42};
    std::uniform_real_distribution<T> Dist{0.0f, 1.0f};
    Mat<BatchSize, Dim, bool> Mask;
  };

  static constexpr size_t InSize = Dim;
  static constexpr size_t OutSize = Dim;

  void forward(const Mat<BatchSize, Dim, T> &X, Mat<BatchSize, Dim, T> &Y) {
    matCopy(X, Y);
  }

  void forward(const Mat<BatchSize, Dim, T> &X, Mat<BatchSize, Dim, T> &Y,
               BackpropAux &Aux) {
    for (size_t I = 0; I < BatchSize; ++I) {
      for (size_t J = 0; J < Dim; ++J) {
        T RandVal = Aux.Dist(Aux.Gen);
        bool Mask = (RandVal >= Config::DropProb);
        Aux.Mask(I, J) = Mask;
        Y(I, J) = Mask ? X(I, J) / (T)(1 - Config::DropProb) : (T)0;
      }
    }
  }

  void backward(const Mat<BatchSize, Dim, T> &DY, Mat<BatchSize, Dim, T> &DX,
                BackpropAux &Aux) {
    matHadamard(DY, Aux.Mask, DX);
    matDivScalar(DX, (T)(1 - Config::DropProb), DX);
  }
};

template <size_t BatchSize, typename T, typename = FloatingPoint<T>>
struct BCEWithLogits {
  Mat<BatchSize, 1, T> Temp1;

  void compute(const Mat<BatchSize, 1, T> &X, const Mat<BatchSize, 1, T> &Y,
               T &Loss, Mat<BatchSize, 1, T> &DX) {
    Mat<BatchSize, 1, T> &Temp2 = DX;
    matApply(X, Temp1, +[](T Val) { return std::max(Val, (T)0); });
    matHadamard(X, Y, Temp2);
    matSub(Temp1, Temp2, Temp1);
    matApply(
        X, Temp2,
        +[](T Val) { return std::log(std::exp(-std::abs(Val)) + (T)1); });
    matAdd(Temp1, Temp2, Temp1);
    matSum(Temp1, Loss);
    Loss /= BatchSize;

    matApply(X, Temp1, sigmoid<T>);
    matSub(Temp1, Y, DX);
    matDivScalar(DX, (T)BatchSize, DX);
  }
};

template <size_t BatchSize, typename T, typename = FloatingPoint<T>>
struct MSE {
  Mat<BatchSize, 1, T> Temp;

  void compute(const Mat<BatchSize, 1, T> &X, const Mat<BatchSize, 1, T> &Y,
               T &Loss, Mat<BatchSize, 1, T> &DX) {
    Mat<BatchSize, 1, T> &Diff = DX;
    matSub(X, Y, Diff);
    matHadamard(Diff, Diff, Temp);
    matSum(Temp, Loss);
    Loss /= BatchSize;

    matHadamardScalar(Diff, (T)2 / (T)BatchSize, DX);
  }
};

template <size_t BatchSize, size_t NumClasses, typename T,
          typename = FloatingPoint<T>>
struct CrossEntropy {
  Mat<BatchSize, 1, T> Max;
  Mat<BatchSize, 1, T> Sum;
  Mat<BatchSize, NumClasses, T> Temp;

  void compute(const Mat<BatchSize, NumClasses, T> &X,
               const Mat<BatchSize, NumClasses, T> &Y, T &Loss,
               Mat<BatchSize, NumClasses, T> &DX) {
    Mat<BatchSize, NumClasses, T> &Exp = DX;
    matMaxRow(X, Max);
    matSubColVec(X, Max, Exp);
    matApply(Exp, Exp, +[](T Val) { return std::exp(Val); });
    matSumRow(Exp, Sum);
    matDivColVec(Exp, Sum, DX);
    matSub(DX, Y, DX);
    matDivScalar(DX, (T)BatchSize, DX);

    matApply(Sum, Sum, +[](T Val) { return std::log(Val); });
    matAdd(Max, Sum, Max);
    matHadamard(Y, X, Temp);
    matSumRow(Temp, Sum);
    matSub(Max, Sum, Max);
    matSum(Max, Loss);
    Loss /= BatchSize;
  }
};

template <typename... Layers> struct LayerList {};

template <size_t BatchSize, typename Container, typename T>
struct Sequential {};

template <size_t BatchSize, typename T, typename... Layers>
struct Sequential<BatchSize, LayerList<Layers...>, T> {
  std::tuple<Layers...> Ls;
  template <typename... Args> struct BufferSelector;

  template <typename Last> struct BufferSelector<Last> {
    using type = std::tuple<>;
  };

  template <typename Current, typename Next, typename... Rest>
  struct BufferSelector<Current, Next, Rest...> {
    using type = decltype(std::tuple_cat(
        std::declval<std::tuple<Mat<BatchSize, Current::OutSize, T>>>(),
        std::declval<typename BufferSelector<Next, Rest...>::type>()));
  };

  using BufferTuple = typename BufferSelector<Layers...>::type;
  BufferTuple Bufs;

  void load(std::ifstream &In) {
    std::apply([&](auto &...L) { (L.load(In), ...); }, Ls);
  }

  template <size_t I = 0, typename InMat, typename OutMat>
  void forwardChain(const InMat &In, OutMat &FinalOut) {
    if constexpr (I == sizeof...(Layers) - 1)
      std::get<I>(Ls).forward(In, FinalOut);
    else {
      auto &L = std::get<I>(Ls);
      auto &NextIn = std::get<I>(Bufs);
      L.forward(In, NextIn);
      forwardChain<I + 1>(NextIn, FinalOut);
    }
  }

  template <typename InMat, typename OutMat>
  void forward(const InMat &X, OutMat &Y) {
    forwardChain<0>(X, Y);
  }
};

template <size_t BatchSize, typename Loss, typename Container, typename T,
          typename OptConfig = DefaultNAdamWConfig<T>>
struct SequentialTrainer {};

template <size_t BatchSize, typename Loss, typename... Layers, typename T,
          typename OptConfig>
struct SequentialTrainer<BatchSize, Loss, LayerList<Layers...>, T, OptConfig>
    : public Sequential<BatchSize, LayerList<Layers...>, T> {
  using Base = Sequential<BatchSize, LayerList<Layers...>, T>;

  std::tuple<typename Layers::BackpropAux...> BackpropAuxs;
  Loss LossFunc;

  using LastLayer =
      std::tuple_element_t<sizeof...(Layers) - 1, std::tuple<Layers...>>;

  static constexpr size_t FinalOutSize = LastLayer::OutSize;
  static constexpr size_t FirstInSize =
      std::tuple_element_t<0, std::tuple<Layers...>>::InSize;

  Mat<BatchSize, FinalOutSize, T> PredY;
  Mat<BatchSize, FinalOutSize, T> DL;
  Mat<BatchSize, FirstInSize, T> TempGrad;

  SequentialTrainer() {
    std::apply([&](auto &...L) { (L.init(), ...); }, this->Ls);
  }

  template <size_t I = 0, typename InMat, typename OutMat>
  void forwardTrainChain(const InMat &In, OutMat &FinalOut) {
    auto &L = std::get<I>(this->Ls);
    auto &BackpropAux = std::get<I>(BackpropAuxs);

    if constexpr (I == sizeof...(Layers) - 1)
      L.forward(In, FinalOut, BackpropAux);
    else {
      auto &NextBuf = std::get<I>(this->Bufs);
      L.forward(In, NextBuf, BackpropAux);
      forwardTrainChain<I + 1>(NextBuf, FinalOut);
    }
  }

  template <size_t I = sizeof...(Layers) - 1, typename GradOutMat,
            typename GradInMat>
  void backwardChain(GradOutMat &GradOut, GradInMat &FinalGradIn) {
    auto &L = std::get<I>(this->Ls);
    auto &BackpropAux = std::get<I>(BackpropAuxs);

    if constexpr (I == 0)
      L.backward(GradOut, FinalGradIn, BackpropAux);
    else {
      auto &PrevLayerGradOut = std::get<I - 1>(this->Bufs);
      L.backward(GradOut, PrevLayerGradOut, BackpropAux);
      backwardChain<I - 1>(PrevLayerGradOut, FinalGradIn);
    }
  }

  template <typename InMat, typename TargetMat>
  void propagate(const InMat &X, const TargetMat &Y, T &LossVal) {
    forwardTrainChain<0>(X, PredY);
    LossFunc.compute(PredY, Y, LossVal, DL);
    backwardChain<sizeof...(Layers) - 1>(DL, TempGrad);
  }

  template <size_t I = 0> void updateChain() {
    auto &L = std::get<I>(this->Ls);
    auto &BackpropAux = std::get<I>(BackpropAuxs);

    if constexpr (I == sizeof...(Layers) - 1) {
      L.template update<OptConfig>(BackpropAux);
    } else {
      L.template update<OptConfig>(BackpropAux);
      updateChain<I + 1>();
    }
  }

  void update() { updateChain<0>(); }

  template <size_t I = 0> void clearGradChain() {
    auto &L = std::get<I>(this->Ls);
    auto &BackpropAux = std::get<I>(BackpropAuxs);

    if constexpr (I == sizeof...(Layers) - 1) {
      L.clearGrad(BackpropAux);
    } else {
      L.clearGrad(BackpropAux);
      clearGradChain<I + 1>();
    }
  }

  void clearGrad() { clearGradChain<0>(); }

  template <size_t I = 0> void scaleGradChain(T Factor) {
    auto &L = std::get<I>(this->Ls);
    auto &BackpropAux = std::get<I>(BackpropAuxs);

    if constexpr (I == sizeof...(Layers) - 1) {
      L.scaleGrad(Factor, BackpropAux);
    } else {
      L.scaleGrad(Factor, BackpropAux);
      scaleGradChain<I + 1>(Factor);
    }
  }

  void scaleGrad(T Factor) { scaleGradChain<0>(Factor); }

  void save(std::ofstream &Out) {
    std::apply([&](auto &...L) { (L.save(Out), ...); }, this->Ls);
  }
};

template <size_t BatchSize, size_t FeatDim, size_t OutDim, typename T,
          typename = FloatingPoint<T>>
struct Batch {
  const Mat<BatchSize, FeatDim, T> &X;
  const Mat<BatchSize, OutDim, T> &Y;
};

template <size_t Size, size_t FeatDim, size_t OutDim, size_t BatchSize,
          typename T, typename = FloatingPoint<T>,
          typename = std::enable_if_t<Size >= BatchSize>>
struct Dataset {
  Mat<Size, FeatDim, T> &X;
  Mat<Size, OutDim, T> &Y;
  std::mt19937_64 Gen;
  std::array<size_t, Size / BatchSize> Indices;

  Dataset(Mat<Size, FeatDim, T> &X, Mat<Size, OutDim, T> &Y, size_t Seed = 42)
      : X(X), Y(Y), Gen(Seed) {
    for (size_t I = 0; I < std::size(Indices); ++I)
      Indices[I] = I * BatchSize;
  }

  void shuffle() { std::shuffle(Indices.begin(), Indices.end(), Gen); }

  struct Iterator {
    friend Dataset;

    Dataset *Parent;
    size_t I;

    Iterator(Dataset *P, size_t I) : Parent(P), I(I) {}

    Batch<BatchSize, FeatDim, OutDim, T> operator*() const {
      const size_t Row = Parent->Indices[I];
      return {Parent->X.template getView<BatchSize>(Row),
              Parent->Y.template getView<BatchSize>(Row)};
    }

    Iterator &operator++() {
      I++;
      return *this;
    }

    Iterator operator++(int) {
      Iterator Temp = *this;
      I++;
      return Temp;
    }

    bool operator!=(const Iterator &Other) const { return I != Other.I; }
  };

  struct Epoch {
    Dataset *Parent;
    Epoch(Dataset *P) : Parent(P) {}
    Iterator begin() { return Iterator{Parent, 0}; }
    Iterator end() { return Iterator{Parent, std::size(Parent->Indices)}; }
  };

  Epoch epoch() {
    shuffle();
    return Epoch{this};
  }

  size_t numBatches() const { return std::size(Indices); }
};

template <size_t NumData, size_t FeatDim, size_t LabelDim, typename T,
          typename U = T, typename = FloatingPoint<T>, typename = Arithmetic<T>>
TINYMLP_ATTRS void shuffle(Mat<NumData, FeatDim, T> &X,
                           Mat<NumData, LabelDim, U> &Y, size_t Seed = 42) {
  std::mt19937_64 Gen(Seed);
  for (size_t I = NumData - 1; I > 0; --I) {
    std::uniform_int_distribution<size_t> Dist(0, I);
    size_t J = Dist(Gen);
    for (size_t K = 0; K < FeatDim; ++K)
      std::swap(X(I, K), X(J, K));
    for (size_t K = 0; K < LabelDim; ++K)
      std::swap(Y(I, K), Y(J, K));
  }
}

template <size_t NumData, typename T, typename U = T,
          typename = FloatingPoint<T>, typename = Arithmetic<U>>
TINYMLP_ATTRS void smoothLabel(Mat<NumData, 1, T> &Y, U Factor) {
  for (size_t I = 0; I < NumData; ++I)
    Y(I, 0) = Y(I, 0) * (1 - Factor) + Factor * (U)0.5;
}

template <size_t NumData, size_t LabelDim, typename T, typename U = T,
          typename = FloatingPoint<T>, typename = Arithmetic<U>>
TINYMLP_ATTRS void smoothLabel(Mat<NumData, LabelDim, T> &Y, U Factor) {
  for (size_t I = 0; I < NumData; ++I)
    for (size_t J = 0; J < LabelDim; ++J)
      Y(I, J) = Y(I, J) * (1 - Factor) + Factor / (U)LabelDim;
}

} // namespace tinymlp

#endif
