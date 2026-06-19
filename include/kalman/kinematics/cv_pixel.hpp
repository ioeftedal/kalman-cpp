#pragma once

#include <Eigen/Core>

namespace kalman::kinematics {

/// 4-state constant-velocity (CV) model in 2-D pixel space.
/// State: [u, v, vu, vv]^T
///
/// F(dt) = [ I₂  dt·I₂ ]
///         [ 0₂   I₂   ]
template <typename T>
constexpr Eigen::Matrix<T, 4, 4> computeF_pixel(T dt) noexcept {
  Eigen::Matrix<T, 4, 4> F = Eigen::Matrix<T, 4, 4>::Identity();
  F(0, 2) = dt;
  F(1, 3) = dt;
  return F;
}

/// Process-noise covariance for the 4-D pixel CV model.
///
/// Q_pp = q_c·dt³/3 · I₂
/// Q_pv = q_c·dt²/2 · I₂
/// Q_vv = q_c·dt    · I₂
template <typename T>
constexpr Eigen::Matrix<T, 4, 4> computeQ_pixel(T dt, T q_c) noexcept {
  T dt2 = dt * dt;
  T dt3 = dt2 * dt;
  T q_pp = q_c * dt3 / T(3);
  T q_pv = q_c * dt2 / T(2);
  T q_vv = q_c * dt;

  Eigen::Matrix<T, 4, 4> Q = Eigen::Matrix<T, 4, 4>::Zero();
  Q(0, 0) = q_pp;
  Q(1, 1) = q_pp;
  Q(0, 2) = q_pv;
  Q(1, 3) = q_pv;
  Q(2, 0) = q_pv;
  Q(3, 1) = q_pv;
  Q(2, 2) = q_vv;
  Q(3, 3) = q_vv;
  return Q;
}

}  // namespace kalman::kinematics
