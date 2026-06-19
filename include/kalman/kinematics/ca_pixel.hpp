#pragma once

#include <Eigen/Core>

namespace kalman::kinematics {

/// 6-state constant-acceleration (CA) model in 2-D pixel space.
/// State: [u, v, vu, vv, au, av]^T
///
/// F(dt) = [ I₂  dt·I₂  ½dt²·I₂ ]
///         [ 0₂   I₂     dt·I₂   ]
///         [ 0₂   0₂      I₂     ]
template <typename T>
constexpr Eigen::Matrix<T, 6, 6> computeF_pixel_CA(T dt) noexcept {
  T dt2 = T(0.5) * dt * dt;
  Eigen::Matrix<T, 6, 6> F = Eigen::Matrix<T, 6, 6>::Identity();
  F(0, 2) = dt;
  F(1, 3) = dt;
  F(0, 4) = dt2;
  F(1, 5) = dt2;
  F(2, 4) = dt;
  F(3, 5) = dt;
  return F;
}

/// Process-noise covariance for the 6-D pixel CA model.
///
/// Q_pp = q_c·dt⁵/20 · I₂
/// Q_pv = q_c·dt⁴/8  · I₂
/// Q_pa = q_c·dt³/6  · I₂
/// Q_vv = q_c·dt³/3  · I₂
/// Q_va = q_c·dt²/2  · I₂
/// Q_aa = q_c·dt     · I₂
template <typename T>
constexpr Eigen::Matrix<T, 6, 6> computeQ_pixel_CA(T dt, T q_c) noexcept {
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

  Eigen::Matrix<T, 6, 6> Q = Eigen::Matrix<T, 6, 6>::Zero();
  Q(0, 0) = q_pp;
  Q(1, 1) = q_pp;
  Q(0, 2) = q_pv;
  Q(1, 3) = q_pv;
  Q(2, 0) = q_pv;
  Q(3, 1) = q_pv;
  Q(0, 4) = q_pa;
  Q(1, 5) = q_pa;
  Q(4, 0) = q_pa;
  Q(5, 1) = q_pa;
  Q(2, 2) = q_vv;
  Q(3, 3) = q_vv;
  Q(2, 4) = q_va;
  Q(3, 5) = q_va;
  Q(4, 2) = q_va;
  Q(5, 3) = q_va;
  Q(4, 4) = q_aa;
  Q(5, 5) = q_aa;
  return Q;
}

}  // namespace kalman::kinematics
