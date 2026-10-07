/// @file
/// @brief Constant-acceleration process model for the interleaved 6-D pixel
///        state `[u, v, vu, vv, au, av]`.
///
/// Continuous-time model: piecewise constant acceleration with white-noise
/// jerk of spectral density `q` (a Wiener process on acceleration).  The
/// discrete transition is exact for constant acceleration; the discrete noise
/// covariance is the exact Gramian of the per-axis triple
///
/// ```
/// F3(dt) = | 1  dt  dt²/2 |        Q3(dt) = q · | dt⁵/20  dt⁴/8   dt³/6 |
///          | 0   1   dt   |                      | dt⁴/8   dt³/3   dt²/2 |
///          | 0   0    1   |                      | dt³/6   dt²/2    dt   |
/// ```
///
/// obtained from `Q = q ∫₀^dt g(τ) g(τ)ᵀ dτ` with `g(τ) = [τ²/2, τ, 1]ᵀ`.
/// The two pixel axes `(u, v)` are independent and are interleaved in the
/// state vector, so the 6-D matrices are assembled by scattering the per-axis
/// blocks into index triples `(a, a+2, a+4)` for `a ∈ {0, 1}`.

#pragma once

#include <Eigen/Core>

namespace kalman::detail {

/// Per-axis CA transition matrix for state `[p, v, a]`.
[[nodiscard]] inline Eigen::Matrix3f caTransitionAxis(float dt) noexcept {
  Eigen::Matrix3f F = Eigen::Matrix3f::Identity();
  F(0, 1) = dt;
  F(0, 2) = 0.5f * dt * dt;
  F(1, 2) = dt;
  return F;
}

/// Per-axis discrete noise covariance for white-jerk spectral density `q`.
[[nodiscard]] inline Eigen::Matrix3f caNoiseAxis(float dt, float q) noexcept {
  const float dt2 = dt * dt;
  const float dt3 = dt2 * dt;
  const float dt4 = dt3 * dt;
  const float dt5 = dt4 * dt;
  Eigen::Matrix3f Q;
  Q << dt5 / 20.0f, dt4 / 8.0f, dt3 / 6.0f,  //
      dt4 / 8.0f, dt3 / 3.0f, dt2 / 2.0f,    //
      dt3 / 6.0f, dt2 / 2.0f, dt;
  return q * Q;
}

/// 6-D CA transition for `[u, v, vu, vv, au, av]`.
[[nodiscard]] inline Eigen::Matrix<float, 6, 6> caTransition(
    float dt) noexcept {
  Eigen::Matrix<float, 6, 6> F = Eigen::Matrix<float, 6, 6>::Identity();
  const Eigen::Matrix3f F3 = caTransitionAxis(dt);
  for (int a = 0; a < 2; ++a) {
    for (int i = 0; i < 3; ++i) {
      for (int j = 0; j < 3; ++j) {
        F(a + 2 * i, a + 2 * j) = F3(i, j);
      }
    }
  }
  return F;
}

/// 6-D discrete noise covariance for white-jerk spectral density `q`.
[[nodiscard]] inline Eigen::Matrix<float, 6, 6> caNoise(float dt,
                                                        float q) noexcept {
  Eigen::Matrix<float, 6, 6> Q = Eigen::Matrix<float, 6, 6>::Zero();
  const Eigen::Matrix3f Q3 = caNoiseAxis(dt, q);
  for (int a = 0; a < 2; ++a) {
    for (int i = 0; i < 3; ++i) {
      for (int j = 0; j < 3; ++j) {
        Q(a + 2 * i, a + 2 * j) = Q3(i, j);
      }
    }
  }
  return Q;
}

}  // namespace kalman::detail
