/// @file
/// @brief 2-D pixel-space tracking models (CV and CA).
///
/// These models operate directly in image pixel coordinates [u, v], where:
///   +u : image right
///   +v : image down

#pragma once

#include <Eigen/Core>

#include "kalman/process_model_fwd.hpp"

namespace kalman {

// ---------------------------------------------------------------------------
//  4-state constant-velocity (CV) model in pixel space
//  x = [u, v, vu, vv]ᵀ
// ---------------------------------------------------------------------------

/// Compute the discrete-time state-transition matrix for the 4-D pixel CV model.
///
/// F(dt) = [ I₂  dt·I₂ ]
///         [ 0₂   I₂   ]
///
/// @param dt  Time step (seconds).
/// @return    4×4 state-transition matrix.
template <typename T>
[[nodiscard]] constexpr Eigen::Matrix<T, 4, 4> computeF_pixel(T dt) noexcept {
  Eigen::Matrix<T, 4, 4> F = Eigen::Matrix<T, 4, 4>::Identity();
  F(0, 2) = dt;
  F(1, 3) = dt;
  return F;
}

/// Compute the discrete-time process-noise covariance for the 4-D pixel CV model.
///
/// Continuous-time process noise is a Wiener process on velocity with PSD q_c:
///   Q_pp = q_c·dt³/3 · I₂
///   Q_pv = q_c·dt²/2 · I₂
///   Q_vv = q_c·dt    · I₂
///
/// @param dt   Time step (seconds).
/// @param q_c  Power spectral density of the Wiener-process velocity (px²/s³).
/// @return     4×4 symmetric positive-definite covariance matrix.
template <typename T>
[[nodiscard]] constexpr Eigen::Matrix<T, 4, 4> computeQ_pixel(T dt, T q_c) noexcept {
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

/// Process model dispatch for the 4-state pixel-space CV model.
template <typename T>
struct ProcessModel<T, 4, void> {
  [[nodiscard]] static constexpr auto F(T dt) noexcept { return computeF_pixel<T>(dt); }
  [[nodiscard]] static constexpr auto Q(T dt, T q_c) noexcept { return computeQ_pixel<T>(dt, q_c); }
};

// ---------------------------------------------------------------------------
//  6-state constant-acceleration (CA) model in pixel space
//  x = [u, v, vu, vv, au, av]ᵀ
// ---------------------------------------------------------------------------

/// Compute the discrete-time state-transition matrix for the 6-D pixel CA model.
///
/// F(dt) = [ I₂  dt·I₂  ½dt²·I₂ ]
///         [ 0₂   I₂     dt·I₂   ]
///         [ 0₂   0₂      I₂     ]
///
/// @param dt  Time step (seconds).
/// @return    6×6 state-transition matrix.
template <typename T>
[[nodiscard]] constexpr Eigen::Matrix<T, 6, 6> computeF_pixel_CA(T dt) noexcept {
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

/// Compute the discrete-time process-noise covariance for the 6-D pixel CA model.
///
/// Continuous-time process noise is a Wiener process on jerk with PSD q_c:
///   Q_pp = q_c·dt⁵/20 · I₂
///   Q_pv = q_c·dt⁴/8  · I₂
///   Q_pa = q_c·dt³/6  · I₂
///   Q_vv = q_c·dt³/3  · I₂
///   Q_va = q_c·dt²/2  · I₂
///   Q_aa = q_c·dt     · I₂
///
/// @param dt   Time step (seconds).
/// @param q_c  Power spectral density of the Wiener-process jerk (px²/s⁵).
/// @return     6×6 symmetric positive-definite covariance matrix.
template <typename T>
[[nodiscard]] constexpr Eigen::Matrix<T, 6, 6> computeQ_pixel_CA(T dt, T q_c) noexcept {
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

/// Tag type used to select the 6-D pixel-space CA process model.
struct PixelCATag {};

/// Process model dispatch for the 6-state pixel-space CA model.
template <typename T>
struct ProcessModel<T, 6, PixelCATag> {
  [[nodiscard]] static constexpr auto F(T dt) noexcept { return computeF_pixel_CA<T>(dt); }
  [[nodiscard]] static constexpr auto Q(T dt, T q_c) noexcept {
    return computeQ_pixel_CA<T>(dt, q_c);
  }
};

// ---------------------------------------------------------------------------
//  Measurement model policy for 2D pixel-space tracking.
//
//  h(x) = [u, v]ᵀ  (linear — extracts the pixel coordinates from the state)
//  No innovation wrapping needed (pixel coords are not angular).
// ---------------------------------------------------------------------------

/// Base policy for 2-D pixel-space tracking.
///
/// @tparam Tag  Model tag to select the process model (void for CV,
///              PixelCATag for CA).
template <typename Tag = void>
struct PixelPolicyBase {
  using ModelTag = Tag;

  /// Measurement model is linear — the UKF can use the closed-form KF
  /// update instead of sigma-point propagation.
  static constexpr bool isMeasurementLinear = true;

  /// Measurement Jacobian — constant for the pixel-extraction model.
  ///
  /// H(x) = [I₂, 0₂, …]  (extracts first two state entries as [u,v]).
  template <typename Derived>
  [[nodiscard]] static constexpr auto H(const Eigen::MatrixBase<Derived>&) noexcept {
    constexpr int N = Derived::RowsAtCompileTime;
    using T = typename Derived::Scalar;
    Eigen::Matrix<T, 2, N> Hmat;
    Hmat.setZero();
    Hmat(0, 0) = T(1);
    Hmat(1, 1) = T(1);
    return Hmat;
  }

  /// Linear measurement function — extracts [u, v] from the state.
  ///
  /// @param x  State vector of dimension >= 2.
  /// @return   [u, v]ᵀ = [x(0), x(1)]ᵀ.
  template <typename Derived>
  [[nodiscard]] static constexpr Eigen::Matrix<typename Derived::Scalar, 2, 1> h(
      const Eigen::MatrixBase<Derived>& x) noexcept {
    static_assert(Derived::RowsAtCompileTime >= 2 && Derived::ColsAtCompileTime == 1);
    Eigen::Matrix<typename Derived::Scalar, 2, 1> z;
    z(0) = x(0);
    z(1) = x(1);
    return z;
  }

  /// Innovation is left unchanged — pixel coordinates are linear and
  /// do not require angular wrapping.
  template <typename T>
  static void wrapInnovation(Eigen::Matrix<T, 2, 1>&) noexcept {
    // no-op
  }
};

/// Predefined policy: 4-state constant-velocity pixel tracking.
using PixelPolicy = PixelPolicyBase<void>;
/// Predefined policy: 6-state constant-acceleration pixel tracking.
using PixelCAPolicy = PixelPolicyBase<PixelCATag>;

}  // namespace kalman
