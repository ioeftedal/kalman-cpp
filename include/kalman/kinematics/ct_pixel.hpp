/// @file
/// @brief Coordinated-turn (CT) motion model in 2-D pixel space.

#pragma once

#include <Eigen/Core>
#include <cmath>

namespace kalman::kinematics {

/// 5-state coordinated-turn (CT) model in 2-D pixel space.
/// State: [u, v, vu, vv, w]^T — `w` is the turn rate (rad/s).
///
/// The model is linear in the state for a frozen turn rate, so the
/// transition matrix is evaluated at the current turn-rate estimate:
///
///                 [ 1   0    sin(w·dt)/w   -(1-cos(w·dt))/w   0 ]
///                 [ 0   1   (1-cos(w·dt))/w   sin(w·dt)/w    0 ]
///   F(w, dt)  =   [ 0   0    cos(w·dt)        -sin(w·dt)     0 ]
///                 [ 0   0    sin(w·dt)        cos(w·dt)      0 ]
///                 [ 0   0    0                0              1 ]
///
/// In the limit w -> 0 the model reduces to the CV model.
template <typename T>
Eigen::Matrix<T, 5, 5> computeF_pixel_CT(const Eigen::Matrix<T, 5, 1>& x, T dt) noexcept {
  const T w = x(4);
  Eigen::Matrix<T, 5, 5> F = Eigen::Matrix<T, 5, 5>::Identity();

  if (std::abs(w) < T(1e-9)) {
    F(0, 2) = dt;
    F(1, 3) = dt;
    return F;
  }

  const T wdt = w * dt;
  const T c = std::cos(wdt);
  const T s = std::sin(wdt);

  F(0, 2) = s / w;
  F(0, 3) = (c - T(1)) / w;
  F(1, 2) = (T(1) - c) / w;
  F(1, 3) = s / w;
  F(2, 2) = c;
  F(2, 3) = -s;
  F(3, 2) = s;
  F(3, 3) = c;
  return F;
}

/// Process-noise covariance for the 5-D pixel CT model.
///
/// A CV block drives position/velocity and an independent term drives the
/// turn-rate state; a full CT coupling would require integrating the F(w,dt)
/// w.r.t. w, which is unnecessary at the IMM's typical operation points.
template <typename T>
constexpr Eigen::Matrix<T, 5, 5> computeQ_pixel_CT(T dt, T q_c) noexcept {
  const T dt2 = dt * dt;
  const T dt3 = dt2 * dt;

  T q_pp = q_c * dt3 / T(3);
  T q_pv = q_c * dt2 / T(2);
  T q_vv = q_c * dt;

  Eigen::Matrix<T, 5, 5> Q = Eigen::Matrix<T, 5, 5>::Zero();
  Q(0, 0) = q_pp;
  Q(1, 1) = q_pp;
  Q(0, 2) = q_pv;
  Q(1, 3) = q_pv;
  Q(2, 0) = q_pv;
  Q(3, 1) = q_pv;
  Q(2, 2) = q_vv;
  Q(3, 3) = q_vv;
  Q(4, 4) = q_c * dt;
  return Q;
}

}  // namespace kalman::kinematics