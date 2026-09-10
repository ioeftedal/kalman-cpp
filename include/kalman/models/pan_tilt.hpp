/// @file
/// @brief 3-D camera-centric pan/tilt tracking models.

#pragma once

#include <Eigen/Core>

#include "kalman/core/constants.hpp"
#include "kalman/core/fwd.hpp"
#include "kalman/kinematics/ca_3d.hpp"
#include "kalman/kinematics/cv_3d.hpp"

namespace kalman {

// Coordinate convention:
//   +x : camera right
//   +y : camera up
//   +z : camera forward (optical axis)

// ---------------------------------------------------------------------------
//  Process model dispatch
// ---------------------------------------------------------------------------

template <typename T>
struct ProcessModel<T, 9, void> {
  static constexpr auto F(T dt) noexcept { return kinematics::computeF_9<T>(dt); }
  static constexpr auto Q(T dt, T q_c) noexcept { return kinematics::computeQ_9<T>(dt, q_c); }
};

template <typename T>
struct ProcessModel<T, 6, void> {
  static constexpr auto F(T dt) noexcept { return kinematics::computeF_6<T>(dt); }
  static constexpr auto Q(T dt, T q_c) noexcept { return kinematics::computeQ_6<T>(dt, q_c); }
};

// ---------------------------------------------------------------------------
//  Non-linear measurement model: pan / tilt from camera-centric position.
//  h(x) = [atan2(x, z),  atan2(y, sqrt(x² + z²))]ᵀ
// ---------------------------------------------------------------------------

template <typename Derived>
constexpr Eigen::Matrix<typename Derived::Scalar, 2, 1> h(
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

struct PanTiltPolicy {
  using ModelTag = void;
  static constexpr bool isMeasurementLinear = false;

  template <typename Derived>
  static constexpr Eigen::Matrix<typename Derived::Scalar, 2, 1> h(
      const Eigen::MatrixBase<Derived>& x) noexcept {
    return kalman::h(x);
  }

  template <typename T>
  static void wrapInnovation(Eigen::Matrix<T, 2, 1>& nu) noexcept {
    constexpr T two_pi = T(2) * constants::pi_v<T>;
    nu(0) = std::remainder(nu(0), two_pi);
  }
};

}  // namespace kalman
