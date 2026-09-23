#pragma once

#include <Eigen/Core>
#include <vector>

#include "kalman/core/constants.hpp"

namespace helpers {

enum class StateComponent { Position, Velocity, Acceleration };

/// Generate a figure-8 Lissajous trajectory for an N-state pixel-space model.
/// Produces true states (noiseless) at given dt for n_steps.
template <int N>
std::vector<Eigen::Matrix<double, N, 1>> generateFigure8States(int n_steps, double dt,
                                                               double amplitude = 500.0,
                                                               double freq = 0.5) {
  std::vector<Eigen::Matrix<double, N, 1>> states;
  states.reserve(n_steps);
  for (int k = 0; k < n_steps; ++k) {
    double t = k * dt;
    Eigen::Matrix<double, N, 1> x;
    x.setZero();
    x(0) = amplitude * std::sin(2 * kalman::constants::pi * freq * t);
    x(1) = amplitude * std::sin(4 * kalman::constants::pi * freq * t);
    if constexpr (N >= 4) {
      x(2) = 2 * kalman::constants::pi * freq * amplitude *
             std::cos(2 * kalman::constants::pi * freq * t);
      x(3) = 4 * kalman::constants::pi * freq * amplitude *
             std::cos(4 * kalman::constants::pi * freq * t);
    }
    if constexpr (N >= 6) {
      double w2 = 2 * kalman::constants::pi * freq;
      double w4 = 4 * kalman::constants::pi * freq;
      x(4) = -w2 * w2 * amplitude * std::sin(w2 * t);
      x(5) = -w4 * w4 * amplitude * std::sin(w4 * t);
    }
    states.push_back(x);
  }
  return states;
}

}  // namespace helpers
