#pragma once

#include <Eigen/Core>

namespace kalman::kinematics {

/// 9-state constant-acceleration (CA) model in 3-D.
/// State: [x, y, z, vx, vy, vz, ax, ay, az]^T
///
/// F(dt) = [ I₃  dt·I₃  ½dt²·I₃ ]
///         [ 0₃   I₃     dt·I₃   ]
///         [ 0₃   0₃      I₃     ]
template <typename T>
constexpr Eigen::Matrix<T, 9, 9> computeF_9(T dt) noexcept {
  Eigen::Matrix<T, 9, 9> F = Eigen::Matrix<T, 9, 9>::Identity();
  F(0, 3) = dt;
  F(1, 4) = dt;
  F(2, 5) = dt;
  F(0, 6) = T(0.5) * dt * dt;
  F(1, 7) = T(0.5) * dt * dt;
  F(2, 8) = T(0.5) * dt * dt;
  F(3, 6) = dt;
  F(4, 7) = dt;
  F(5, 8) = dt;
  return F;
}

/// Process-noise covariance for the 9-D CA model.
///
/// Q_pp = q_c·dt⁵/20 · I₃
/// Q_pv = q_c·dt⁴/8  · I₃
/// Q_pa = q_c·dt³/6  · I₃
/// Q_vv = q_c·dt³/3  · I₃
/// Q_va = q_c·dt²/2  · I₃
/// Q_aa = q_c·dt     · I₃
template <typename T>
constexpr Eigen::Matrix<T, 9, 9> computeQ_9(T dt, T q_c) noexcept {
  T dt2 = dt * dt;
  T dt3 = dt2 * dt;
  T dt4 = dt3 * dt;
  T dt5 = dt4 * dt;

  T q_pp = q_c * dt5 / T(20);
  T q_pv = q_c * dt4 / T(8);
  T q_pa = q_c * dt3 / T(6);
  T q_vv = q_c * dt3 / T(3);
  T q_va = q_c * dt2 / T(2);
  T q_aa = q_c * dt;

  Eigen::Matrix<T, 9, 9> Q = Eigen::Matrix<T, 9, 9>::Zero();
  Q(0, 0) = q_pp;
  Q(1, 1) = q_pp;
  Q(2, 2) = q_pp;
  Q(0, 3) = q_pv;
  Q(1, 4) = q_pv;
  Q(2, 5) = q_pv;
  Q(3, 0) = q_pv;
  Q(4, 1) = q_pv;
  Q(5, 2) = q_pv;
  Q(0, 6) = q_pa;
  Q(1, 7) = q_pa;
  Q(2, 8) = q_pa;
  Q(6, 0) = q_pa;
  Q(7, 1) = q_pa;
  Q(8, 2) = q_pa;
  Q(3, 3) = q_vv;
  Q(4, 4) = q_vv;
  Q(5, 5) = q_vv;
  Q(3, 6) = q_va;
  Q(4, 7) = q_va;
  Q(5, 8) = q_va;
  Q(6, 3) = q_va;
  Q(7, 4) = q_va;
  Q(8, 5) = q_va;
  Q(6, 6) = q_aa;
  Q(7, 7) = q_aa;
  Q(8, 8) = q_aa;
  return Q;
}

}  // namespace kalman::kinematics
