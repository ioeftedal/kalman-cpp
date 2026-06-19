/// @file
/// @brief 1-D thermal CV UKF tests — example of a minimal model test file.

#include <gtest/gtest.h>

#include <cmath>
#include <numbers>

#include "../helpers/model_test.hpp"
#include "../helpers/trajectories.hpp"
#include "kalman/core/filter.hpp"
#include "kalman/models/thermal.hpp"

// -----------------------------------------------------------------------
//  Type alias
// -----------------------------------------------------------------------

using ThermalFilter = kalman::UnscentedKalmanFilter<double, 2, 1, kalman::ThermalPolicy>;

// -----------------------------------------------------------------------
//  Process-model structure
// -----------------------------------------------------------------------

TEST(ThermalProcessModel, FStructure) { helpers::testProcessModelStructure<ThermalFilter>(100.0); }

// -----------------------------------------------------------------------
//  Numerical stability
// -----------------------------------------------------------------------

TEST(ThermalNumericalStability, SmallDt) { helpers::testNumericalStability<ThermalFilter>(100.0); }

// -----------------------------------------------------------------------
//  Convergence — sinusoidal temperature trajectory
// -----------------------------------------------------------------------

namespace {

std::vector<Eigen::Matrix<double, 2, 1>> generateThermalTrajectory(int n_steps, double dt,
                                                                   double T0 = 25.0,
                                                                   double amp = 10.0,
                                                                   double freq = 0.2) {
  std::vector<Eigen::Matrix<double, 2, 1>> states;
  states.reserve(n_steps);
  for (int k = 0; k < n_steps; ++k) {
    double t = k * dt;
    Eigen::Matrix<double, 2, 1> x;
    double w = 2 * std::numbers::pi * freq;
    x(0) = T0 + amp * std::sin(w * t);  // temperature
    x(1) = amp * w * std::cos(w * t);   // dT/dt
    states.push_back(x);
  }
  return states;
}

}  // namespace

TEST(ThermalConvergence, SinusoidalTemperature) {
  auto states = generateThermalTrajectory(200, 1.0 / 10, 25.0, 10.0, 0.2);
  // N=2 → chi2_95 = 5.99
  helpers::testConvergence<ThermalFilter>(states, 100.0, 1.0 / 10, 1.0, 5.99, 20.0, 10.0);
}

// -----------------------------------------------------------------------
//  Monte Carlo NEES consistency
// -----------------------------------------------------------------------

TEST(ThermalNEES, MonteCarloConsistency) {
  auto states = generateThermalTrajectory(100, 1.0 / 10, 25.0, 10.0, 0.2);
  // N=2, 30 runs: chi2_0.05(30*2) ≈ 1.0, chi2_0.95(30*2) ≈ 3.0 (scaled by 2N)
  helpers::testNEESConsistency<ThermalFilter>(states, 100.0, 1.0 / 10, 1.0, 1.0, 7.0, 30, 20.0,
                                              10.0);
}
