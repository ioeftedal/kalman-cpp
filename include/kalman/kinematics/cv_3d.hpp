#pragma once

#include <Eigen/Core>

namespace kalman::kinematics {

/// 6-state constant-velocity (CV) model in 3-D.
/// State: [x, y, z, vx, vy, vz]^T
///
/// F(dt) = [ I₃  dt·I₃ ]
///         [ 0₃   I₃   ]
template <typename T>
constexpr Eigen::Matrix<T, 6, 6> computeF_6(T dt) noexcept {
  Eigen::Matrix<T, 6, 6> F = Eigen::Matrix<T, 6, 6>::Identity();
  F(0, 3) = dt;
  F(1, 4) = dt;
  F(2, 5) = dt;
  return F;
}

/// Process-noise covariance for the 6-D CV model.
///
/// Q_pp = q_c·dt³/3 · I₃
/// Q_pv = q_c·dt²/2 · I₃
/// Q_vv = q_c·dt    · I₃
template <typename T>
constexpr Eigen::Matrix<T, 6, 6> computeQ_6(T dt, T q_c) noexcept {
  T dt2 = dt * dt;
  T dt3 = dt2 * dt;
  T q_pp = q_c * dt3 / T(3);
  T q_pv = q_c * dt2 / T(2);
  T q_vv = q_c * dt;

  Eigen::Matrix<T, 6, 6> Q = Eigen::Matrix<T, 6, 6>::Zero();
  Q(0, 0) = q_pp;
  Q(1, 1) = q_pp;
  Q(2, 2) = q_pp;
  Q(0, 3) = q_pv;
  Q(1, 4) = q_pv;
  Q(2, 5) = q_pv;
  Q(3, 0) = q_pv;
  Q(4, 1) = q_pv;
  Q(5, 2) = q_pv;
  Q(3, 3) = q_vv;
  Q(4, 4) = q_vv;
  Q(5, 5) = q_vv;
  return Q;
}

}  // namespace kalman::kinematics
