/// @file
/// @brief 3-D camera-centric pan/tilt tracking models.
///
/// Provides the process model (constant-acceleration and constant-velocity)
/// and the non-linear bearing-only measurement model for tracking in camera
/// coordinates.

#pragma once

#include <Eigen/Core>
#include <numbers>

#include "kalman/process_model_fwd.hpp"

namespace kalman {

// Coordinate convention:
//   +x : camera right
//   +y : camera up
//   +z : camera forward (optical axis)

// ---------------------------------------------------------------------------
//  9-state constant-acceleration (CA) model
//  x = [x, y, z, vx, vy, vz, ax, ay, az]ᵀ
// ---------------------------------------------------------------------------

/// Compute the discrete-time state-transition matrix for the 9-D CA model.
///
/// The continuous-time model is:
///   d/dt [p, v, a]ᵀ = [v, a, w]ᵀ,   w ~ Wiener process (jerk)
///
/// Exact discretisation yields:
///   F(dt) = [ I₃  dt·I₃  ½dt²·I₃ ]
///           [ 0₃   I₃     dt·I₃   ]
///           [ 0₃   0₃      I₃     ]
///
/// @param dt  Time step (seconds).
/// @return    9×9 state-transition matrix.
template <typename T>
[[nodiscard]] constexpr Eigen::Matrix<T, 9, 9> computeF_9(T dt) noexcept {
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

/// Compute the discrete-time process-noise covariance for the 9-D CA model.
///
/// The continuous-time process noise is a Wiener process on jerk with PSD q_c.
/// Exact discretisation yields the block-matrix elements:
///   Q_pp = q_c·dt⁵/20 · I₃    (position–position)
///   Q_pv = q_c·dt⁴/8  · I₃    (position–velocity)
///   Q_pa = q_c·dt³/6  · I₃    (position–acceleration)
///   Q_vv = q_c·dt³/3  · I₃    (velocity–velocity)
///   Q_va = q_c·dt²/2  · I₃    (velocity–acceleration)
///   Q_aa = q_c·dt     · I₃    (acceleration–acceleration)
///
/// @param dt   Time step (seconds).
/// @param q_c  Power spectral density of the Wiener-process jerk.
/// @return     9×9 symmetric positive-definite process-noise covariance.
template <typename T>
[[nodiscard]] constexpr Eigen::Matrix<T, 9, 9> computeQ_9(T dt, T q_c) noexcept {
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

// ---------------------------------------------------------------------------
//  6-state constant-velocity (CV) model
//  x = [x, y, z, vx, vy, vz]ᵀ
// ---------------------------------------------------------------------------

/// Compute the discrete-time state-transition matrix for the 6-D CV model.
///
/// F(dt) = [ I₃  dt·I₃ ]
///         [ 0₃   I₃   ]
///
/// @param dt  Time step (seconds).
/// @return    6×6 state-transition matrix.
template <typename T>
[[nodiscard]] constexpr Eigen::Matrix<T, 6, 6> computeF_6(T dt) noexcept {
  Eigen::Matrix<T, 6, 6> F = Eigen::Matrix<T, 6, 6>::Identity();
  F(0, 3) = dt;
  F(1, 4) = dt;
  F(2, 5) = dt;
  return F;
}

/// Compute the discrete-time process-noise covariance for the 6-D CV model.
///
/// Continuous-time process noise is a Wiener process on velocity with PSD q_c:
///   Q_pp = q_c·dt³/3 · I₃
///   Q_pv = q_c·dt²/2 · I₃
///   Q_vv = q_c·dt    · I₃
///
/// @param dt   Time step (seconds).
/// @param q_c  Power spectral density of the Wiener-process velocity.
/// @return     6×6 symmetric positive-definite process-noise covariance.
template <typename T>
[[nodiscard]] constexpr Eigen::Matrix<T, 6, 6> computeQ_6(T dt, T q_c) noexcept {
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

// ---------------------------------------------------------------------------
//  Process model dispatch — selects CA (N=9) or CV (N=6) at compile time
// ---------------------------------------------------------------------------

/// Process model for N=9, default tag — constant acceleration.
template <typename T>
struct ProcessModel<T, 9, void> {
  [[nodiscard]] static constexpr auto F(T dt) noexcept { return computeF_9<T>(dt); }
  [[nodiscard]] static constexpr auto Q(T dt, T q_c) noexcept { return computeQ_9<T>(dt, q_c); }
};

/// Process model for N=6, default tag — constant velocity.
template <typename T>
struct ProcessModel<T, 6, void> {
  [[nodiscard]] static constexpr auto F(T dt) noexcept { return computeF_6<T>(dt); }
  [[nodiscard]] static constexpr auto Q(T dt, T q_c) noexcept { return computeQ_6<T>(dt, q_c); }
};

// ---------------------------------------------------------------------------
//  Non-linear measurement model: pan / tilt from camera-centric position.
//  Works with any state vector whose first 3 elements are [x, y, z].
//
//  h(x) = [atan2(x, z),  atan2(y, sqrt(x² + z²))]ᵀ
// ---------------------------------------------------------------------------

/// Non-linear bearing-only measurement function.
///
/// Given a camera-centric position [x, y, z] in the first three state
/// elements, computes:
///   pan  = atan2(x, z)               — horizontal bearing (radians)
///   tilt = atan2(y, √(x² + z²))     — vertical bearing   (radians)
///
/// @tparam Derived  Eigen vector type (must have rows 6 or 9).
/// @param x         State vector whose first 3 entries are [x, y, z].
/// @return          2×1 measurement [pan, tilt]ᵀ.
template <typename Derived>
[[nodiscard]] constexpr Eigen::Matrix<typename Derived::Scalar, 2, 1> h(
    const Eigen::MatrixBase<Derived>& x) noexcept {
  static_assert((Derived::RowsAtCompileTime == 6 || Derived::RowsAtCompileTime == 9) &&
                Derived::ColsAtCompileTime == 1);
  using T = typename Derived::Scalar;
  using std::atan2;
  using std::sqrt;
  Eigen::Matrix<T, 2, 1> z;
  z(0) = atan2(x(0), x(2));
  z(1) = atan2(x(1), sqrt(x(0) * x(0) + x(2) * x(2)));
  return z;
}

// ---------------------------------------------------------------------------
//  Measurement model policy for 3D pan/tilt bearing-only tracking.
// ---------------------------------------------------------------------------

/// Policy struct for 3-D pan/tilt bearing-only tracking with the UKF.
///
/// Provides:
///   - `ModelTag` (void) — selects the default (N=9, N=6) process models.
///   - `h(x)` — delegates to the free function `kalman::h(x)`.
///   - `wrapInnovation(nu)` — wraps the innovation vector so that the pan
///     component lies in the interval (-π, π], preventing large spurious
///     innovations caused by angular wraparound.
struct PanTiltPolicy {
  using ModelTag = void;
  static constexpr bool isMeasurementLinear = false;

  /// Measurement function: pan/tilt from camera-centric position.
  template <typename Derived>
  [[nodiscard]] static constexpr Eigen::Matrix<typename Derived::Scalar, 2, 1> h(
      const Eigen::MatrixBase<Derived>& x) noexcept {
    static_assert((Derived::RowsAtCompileTime == 6 || Derived::RowsAtCompileTime == 9) &&
                  Derived::ColsAtCompileTime == 1);
    return kalman::h(x);
  }

  /// Wrap the innovation such that the pan component lies in (-π, π].
  ///
  /// The tilt component is not wrapped — it is naturally bounded to ±π/2 by
  /// the atan2 definition.
  template <typename T>
  static void wrapInnovation(Eigen::Matrix<T, 2, 1>& nu) noexcept {
    constexpr T two_pi = T(2) * std::numbers::pi_v<T>;
    nu(0) = std::remainder(nu(0), two_pi);
  }
};

}  // namespace kalman
