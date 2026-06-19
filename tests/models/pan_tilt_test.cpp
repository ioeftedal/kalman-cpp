/// @file
/// @brief 3-D pan/tilt UKF tests (bearing-only non-linear model).

#include "kalman/models/pan_tilt.hpp"

#include <gtest/gtest.h>

#include <cmath>
#include <numbers>

#include "../helpers/model_test.hpp"
#include "kalman/core/filter.hpp"

// -----------------------------------------------------------------------
//  Type aliases
// -----------------------------------------------------------------------

using CaFilter = kalman::UnscentedKalmanFilter<double, 9, 2, kalman::PanTiltPolicy>;
using CvFilter = kalman::UnscentedKalmanFilter<double, 6, 2, kalman::PanTiltPolicy>;

// -----------------------------------------------------------------------
//  Process-model structure tests
// -----------------------------------------------------------------------

TEST(PanTiltProcessModel, CA_FStructure) { helpers::testProcessModelStructure<CaFilter>(500.0); }

TEST(PanTiltProcessModel, CV_FStructure) { helpers::testProcessModelStructure<CvFilter>(500.0); }

// -----------------------------------------------------------------------
//  Numerical stability
// -----------------------------------------------------------------------

TEST(PanTiltNumericalStability, CA_SmallDt) { helpers::testNumericalStability<CaFilter>(500.0); }

TEST(PanTiltNumericalStability, CV_SmallDt) { helpers::testNumericalStability<CvFilter>(500.0); }

// -----------------------------------------------------------------------
//  Measurement model — known positions
// -----------------------------------------------------------------------

TEST(PanTiltMeasurement, KnownPositions) {
  using Vec6 = Eigen::Matrix<double, 6, 1>;

  // Target directly ahead → pan=0, tilt=0
  auto z1 = kalman::h(Vec6{0, 0, 100, 0, 0, 0});
  EXPECT_NEAR(z1(0), 0.0, 1e-10);
  EXPECT_NEAR(z1(1), 0.0, 1e-10);

  // Target to the right → positive pan
  auto z2 = kalman::h(Vec6{100, 0, 0, 0, 0, 0});
  EXPECT_NEAR(z2(0), std::numbers::pi / 2, 1e-10);
  EXPECT_NEAR(z2(1), 0.0, 1e-10);

  // Target above → positive tilt
  auto z3 = kalman::h(Vec6{0, 100, 100, 0, 0, 0});
  EXPECT_NEAR(z3(0), 0.0, 1e-10);
  EXPECT_NEAR(z3(1), std::numbers::pi / 4, 1e-10);

  // Behind → pan = ±π
  auto z4 = kalman::h(Vec6{0, 0, -100, 0, 0, 0});
  EXPECT_NEAR(std::abs(z4(0)), std::numbers::pi, 1e-10);
  EXPECT_NEAR(z4(1), 0.0, 1e-10);
}

// -----------------------------------------------------------------------
//  Angle wrapping
// -----------------------------------------------------------------------

TEST(PanTiltMeasurement, InnovationWrapping) {
  Eigen::Vector2d nu;
  nu << 3.0, 0.5;  // pan outside (-π, π]
  kalman::PanTiltPolicy::wrapInnovation(nu);
  EXPECT_GT(nu(0), -std::numbers::pi);
  EXPECT_LE(nu(0), std::numbers::pi);
  EXPECT_NEAR(nu(1), 0.5, 1e-10);  // tilt unchanged
}

// -----------------------------------------------------------------------
//  Convergence — constant-acceleration trajectory
// -----------------------------------------------------------------------

namespace {

std::vector<Eigen::Matrix<double, 9, 1>> generateConstantAccelTrajectory(int n_steps, double dt) {
  std::vector<Eigen::Matrix<double, 9, 1>> states;
  states.reserve(n_steps);

  // Initial state: moving along +z (optical axis), accelerating
  Eigen::Matrix<double, 9, 1> x;
  x << 5.0, 3.0, 50.0,  // position [x, y, z]
      0.5, 0.2, 10.0,   // velocity
      0.0, 0.0, 1.0;    // acceleration

  for (int k = 0; k < n_steps; ++k) {
    states.push_back(x);
    // Exact integration: x += v*dt + 0.5*a*dt², v += a*dt
    x(0) += x(3) * dt + 0.5 * x(6) * dt * dt;
    x(1) += x(4) * dt + 0.5 * x(7) * dt * dt;
    x(2) += x(5) * dt + 0.5 * x(8) * dt * dt;
    x(3) += x(6) * dt;
    x(4) += x(7) * dt;
    x(5) += x(8) * dt;
  }
  return states;
}

}  // namespace

TEST(PanTiltConvergence, ConstantAccelerationTrajectory) {
  auto states = generateConstantAccelTrajectory(100, 1.0 / 30);
  helpers::testConvergence<CaFilter>(states, 500.0, 1.0 / 30, 0.05, 12.59);
}

// -----------------------------------------------------------------------
//  Monte Carlo NEES consistency
// -----------------------------------------------------------------------

TEST(PanTiltNEES, MonteCarloConsistency) {
  auto states = generateConstantAccelTrajectory(80, 1.0 / 30);
  helpers::testNEESConsistency<CaFilter>(states, 500.0, 1.0 / 30, 0.05, 5.90, 21.92, 30);
}
