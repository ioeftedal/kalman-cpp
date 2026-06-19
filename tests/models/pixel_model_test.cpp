/// @file
/// @brief 2-D pixel-space UKF tests (CV and CA models).

#include <gtest/gtest.h>

#include <cmath>
#include <numbers>

#include "../helpers/model_test.hpp"
#include "../helpers/trajectories.hpp"
#include "kalman/core/filter.hpp"
#include "kalman/models/pixel.hpp"

// -----------------------------------------------------------------------
//  Type aliases
// -----------------------------------------------------------------------

using PixelCvFilter = kalman::UnscentedKalmanFilter<double, 4, 2, kalman::PixelPolicy>;
using PixelCaFilter = kalman::UnscentedKalmanFilter<double, 6, 2, kalman::PixelCAPolicy>;

// -----------------------------------------------------------------------
//  Process-model structure tests
// -----------------------------------------------------------------------

TEST(PixelCVProcessModel, FStructure) { helpers::testProcessModelStructure<PixelCvFilter>(500.0); }

TEST(PixelCAProcessModel, FStructure) { helpers::testProcessModelStructure<PixelCaFilter>(500.0); }

// -----------------------------------------------------------------------
//  Numerical stability tests
// -----------------------------------------------------------------------

TEST(PixelCVNumericalStability, SmallDt) { helpers::testNumericalStability<PixelCvFilter>(500.0); }

TEST(PixelCANumericalStability, SmallDt) { helpers::testNumericalStability<PixelCaFilter>(500.0); }

// -----------------------------------------------------------------------
//  Convergence tests (figure-8 Lissajous trajectory)
// -----------------------------------------------------------------------

// Slow figure-8 so the CV model (constant-velocity) can track it with
// modest process noise.  Constraint: peak acceleration² × dt ≪ (qc · dt).
// With amp=200, freq=0.1 → ω=0.628 rad/s → accel = 200·ω² ≈ 79 px/s²,
// so qc ~ 5000 gives Qvel = qc·dt ≈ 167 (σ ≈ 13 px/s), well above the
// 2.6 px/s velocity change per step.
constexpr double kFig8Amp = 200.0;
constexpr double kFig8Freq = 0.1;
constexpr int kFig8Steps = 200;
constexpr double kFig8Dt = 1.0 / 30;

TEST(PixelCVConvergence, Figure8Trajectory) {
  auto states = helpers::generateFigure8States<4>(kFig8Steps, kFig8Dt, kFig8Amp, kFig8Freq);
  helpers::testConvergence<PixelCvFilter>(states, 5000.0, kFig8Dt, 5.0, 9.49, 100.0, 50.0);
}

TEST(PixelCAConvergence, Figure8Trajectory) {
  auto states = helpers::generateFigure8States<6>(kFig8Steps, kFig8Dt, kFig8Amp, kFig8Freq);
  helpers::testConvergence<PixelCaFilter>(states, 5000.0, kFig8Dt, 5.0, 12.59, 100.0, 50.0);
}

// -----------------------------------------------------------------------
//  Monte Carlo NEES consistency
// -----------------------------------------------------------------------

TEST(PixelCVNEES, MonteCarloConsistency) {
  auto states = helpers::generateFigure8States<4>(kFig8Steps, kFig8Dt, kFig8Amp, kFig8Freq);
  helpers::testNEESConsistency<PixelCvFilter>(states, 5000.0, kFig8Dt, 5.0, 2.0, 7.0, 50, 100.0,
                                              50.0);
}

TEST(PixelCANEES, MonteCarloConsistency) {
  auto states = helpers::generateFigure8States<6>(kFig8Steps, kFig8Dt, kFig8Amp, kFig8Freq);
  helpers::testNEESConsistency<PixelCaFilter>(states, 5000.0, kFig8Dt, 5.0, 3.0, 10.0, 30, 100.0,
                                              50.0);
}
