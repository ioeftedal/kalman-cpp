/// @file
/// @brief Allocation-freedom tests for the filter hot path, with Eigen's
///        internal allocator visible.
///
/// These filters are embedded in fixed-period control loops with a hard
/// deadline and an explicit "no heap allocation on the hot path" budget. The
/// tests below bracket a run of predict/update cycles and assert the
/// allocation delta is exactly zero.
///
/// This target is compiled with `KALMAN_TEST_WRAP_MALLOC` and linked with the
/// linker's `--wrap` options so that `malloc` -- which is how Eigen's
/// `aligned_malloc` actually allocates -- is counted. A plain `operator new`
/// hook cannot see Eigen at all; see alloc_counter.hpp and the EigenVisibility
/// tests below, which pin that distinction down.

#include <gtest/gtest.h>

#include <Eigen/Core>
#include <cmath>

#include "helpers/alloc_counter.hpp"
#include "kalman/core/imm.hpp"
#include "kalman/models/pixel.hpp"

// ---------------------------------------------------------------------------
//  Types -- instantiated with float, matching the aarch64 consumers that use
//  this library (no hardware FP64).
// ---------------------------------------------------------------------------

using ImmCvModelF = kalman::ImmPixelModel<float, void>;
using ImmCaModelF = kalman::ImmPixelModel<float, kalman::PixelCATag>;
using ImmCtModelF = kalman::ImmPixelModel<float, kalman::PixelCTTag>;
using ImmF = kalman::ImmFilter<float, ImmCvModelF, ImmCaModelF, ImmCtModelF>;

using PixelCvF = kalman::UnscentedKalmanFilter<float, 4, 2, kalman::PixelPolicy>;

namespace {

constexpr int kWarmup = 32;
constexpr int kIters = 512;
constexpr float kDt = 1.0F / 30.0F;

ImmF::VectorC initialState() {
  ImmF::VectorC x;
  x << 320.0F, 240.0F, 12.0F, -7.0F;
  return x;
}

ImmF::MatrixC initialCov() {
  ImmF::MatrixC P = ImmF::MatrixC::Identity() * (20.0F * 20.0F);
  P(2, 2) = 15.0F * 15.0F;
  P(3, 3) = 15.0F * 15.0F;
  return P;
}

ImmF makeImm(bool adaptive) {
  ImmF imm;
  imm.setTransitionProbabilities(0.02F);
  imm.setProcessNoiseSpectralDensities({2000.0F, 4000.0F, 2500.0F});
  imm.setAdaptiveNoise(adaptive, 30, 2.0F);
  imm.initialize(initialState(), initialCov());
  return imm;
}

/// Drive `iters` predict+update cycles. Kept in its own function so no
/// test-framework bookkeeping lands inside the measured region.
void runCycles(ImmF& imm, int iters) {
  ImmF::MatrixM R = ImmF::MatrixM::Identity() * (4.0F * 4.0F);
  const ImmF::VectorC x0 = initialState();
  for (int i = 0; i < iters; ++i) {
    imm.predict(kDt);
    ImmF::VectorM z;
    z(0) = x0(0) + 18.0F * static_cast<float>(i) + static_cast<float>(i % 3);
    z(1) = x0(1) - 9.0F * static_cast<float>(i);
    imm.update(z, R, 9.21F);
  }
}

}  // namespace

// ---------------------------------------------------------------------------
//  Eigen visibility -- pins down that this binary really can see Eigen
//  allocation, so the zero-delta expectations below are not vacuous.
// ---------------------------------------------------------------------------

// The matrix must be heap-allocated and escape: a small non-escaping local of
// dynamic size is scalar-replaced by the optimiser and never reaches
// aligned_malloc at all.
Eigen::Matrix<double, 2, Eigen::Dynamic>* volatile g_sink = nullptr;

TEST(EigenVisibility, DynamicMatrixAllocationIsObserved) {
  alloc_counter::reset();
  const alloc_counter::Scope scope;
  {
    auto* gain = new Eigen::Matrix<double, 2, Eigen::Dynamic>(2, 6);
    gain->setZero();
    (*gain)(0, 0) = 1.0;
    (*gain)(1, 1) = 1.0;
    g_sink = gain;
    EXPECT_EQ(gain->rows(), 2);
    EXPECT_EQ(gain->cols(), 6);
    EXPECT_DOUBLE_EQ((*gain)(1, 1), 1.0);
  }
  EXPECT_GE(scope.delta(), 1U)
      << "malloc interposition is not active: Eigen allocation is invisible, so the "
         "no-allocation tests in this binary would pass vacuously";
  delete g_sink;
  g_sink = nullptr;
}

TEST(EigenVisibility, FixedSizeMatrixOperationDoesNotAllocate) {
  // Confirms the detector is not simply counting every Eigen call.
  Eigen::Matrix4d m = Eigen::Matrix4d::Zero();
  m(0, 0) = 1.0;
  alloc_counter::reset();
  const alloc_counter::Scope scope;
  const volatile double s = (m * m.transpose()).sum();
  EXPECT_EQ(scope.delta(), 0U);
  EXPECT_DOUBLE_EQ(s, 1.0);
}

// ---------------------------------------------------------------------------
//  Allocation freedom of the filter hot path
// ---------------------------------------------------------------------------

TEST(ImmNoAlloc, PredictAndUpdateAreAllocationFree) {
  ImmF imm = makeImm(/*adaptive=*/true);
  runCycles(imm, kWarmup);  // warm any lazy one-time initialisation

  alloc_counter::reset();
  runCycles(imm, kIters);

  EXPECT_EQ(alloc_counter::count(), 0U)
      << "ImmFilter::predict/update must not allocate on the hot path (ran " << kIters
      << " cycles)";
}

TEST(ImmNoAlloc, UpdateVelocityAndSetPositionAreAllocationFree) {
  ImmF imm = makeImm(/*adaptive=*/true);
  runCycles(imm, kWarmup);

  ImmF::MatrixM R = ImmF::MatrixM::Identity() * (4.0F * 4.0F);
  const ImmF::VectorM z_vel(18.0F, -9.0F);
  const ImmF::VectorM z_pos(340.0F, 232.0F);

  alloc_counter::reset();
  for (int i = 0; i < kIters; ++i) {
    imm.updateVelocity(z_vel, R);
    imm.setPosition(z_pos, R);
  }

  EXPECT_EQ(alloc_counter::count(), 0U)
      << "ImmFilter::updateVelocity/setPosition must not allocate";
}

TEST(ImmNoAlloc, SinglePixelFilterIsAllocationFree) {
  PixelCvF f(1.0F, 2.0F, 0.0F);
  f.setProcessNoiseSpectralDensity(500.0F);
  f.setMeasurementNoiseCovariance(PixelCvF::MatrixM::Identity() * (4.0F * 4.0F));

  Eigen::Matrix<float, 4, 1> x0;
  x0 << 320.0F, 240.0F, 12.0F, -7.0F;
  f.setState(x0);
  f.setCovariance(Eigen::Matrix<float, 4, 4>::Identity() * 400.0F);

  for (int i = 0; i < kWarmup; ++i) {
    f.filter(kDt, Eigen::Matrix<float, 2, 1>(340.0F, 236.0F));
  }

  alloc_counter::reset();
  for (int i = 0; i < kIters; ++i) {
    const Eigen::Matrix<float, 2, 1> z(340.0F + static_cast<float>(i % 5), 236.0F);
    f.filter(kDt, z);
  }

  EXPECT_EQ(alloc_counter::count(), 0U)
      << "UnscentedKalmanFilter::filter must not allocate on the hot path";
}

TEST(ImmNoAlloc, WrappedAdaptiveHistoryIsAllocationFree) {
  // Drive well past the ring-buffer capacity so the wrap-around path of the
  // running mean is exercised, not just the first fill.
  ImmF imm = makeImm(/*adaptive=*/true);
  imm.setAdaptiveNoise(true, ImmF::kMaxAdaptWindow, 2.0F);
  runCycles(imm, kWarmup);

  const int iters = ImmF::kMaxAdaptWindow + 200;
  alloc_counter::reset();
  runCycles(imm, iters);

  EXPECT_EQ(alloc_counter::count(), 0U) << "wrapped ring buffer must not allocate";
  EXPECT_TRUE(std::isfinite(imm.noiseScale()));
  EXPECT_GT(imm.noiseScale(), 0.0F);
}

TEST(ImmNoAlloc, OverLargeAdaptiveWindowIsClampedNotAllocated) {
  ImmF imm = makeImm(/*adaptive=*/true);
  imm.setAdaptiveNoise(true, /*window=*/100000, 2.0F);
  runCycles(imm, kWarmup);

  alloc_counter::reset();
  runCycles(imm, 256);

  EXPECT_EQ(alloc_counter::count(), 0U)
      << "an over-large window must be clamped to kMaxAdaptWindow, not allocated";
  EXPECT_TRUE(std::isfinite(imm.noiseScale()));
}