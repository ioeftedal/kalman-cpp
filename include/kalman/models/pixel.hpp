/// @file
/// @brief 2-D pixel-space tracking models (CV and CA).

#pragma once

#include <Eigen/Core>

#include "kalman/core/fwd.hpp"
#include "kalman/kinematics/ca_pixel.hpp"
#include "kalman/kinematics/cv_pixel.hpp"

namespace kalman {

// ---------------------------------------------------------------------------
//  Process model dispatch
// ---------------------------------------------------------------------------

template <typename T>
struct ProcessModel<T, 4, void> {
  static constexpr auto F(T dt) noexcept { return kinematics::computeF_pixel<T>(dt); }
  static constexpr auto Q(T dt, T q_c) noexcept { return kinematics::computeQ_pixel<T>(dt, q_c); }
};

struct PixelCATag {};

template <typename T>
struct ProcessModel<T, 6, PixelCATag> {
  static constexpr auto F(T dt) noexcept { return kinematics::computeF_pixel_CA<T>(dt); }
  static constexpr auto Q(T dt, T q_c) noexcept {
    return kinematics::computeQ_pixel_CA<T>(dt, q_c);
  }
};

// ---------------------------------------------------------------------------
//  Measurement model policy for 2D pixel-space tracking.
// ---------------------------------------------------------------------------

template <typename Tag = void>
struct PixelPolicyBase {
  using ModelTag = Tag;
  static constexpr bool isMeasurementLinear = true;

  template <typename Derived>
  static constexpr auto H(const Eigen::MatrixBase<Derived>&) noexcept {
    constexpr int N = Derived::RowsAtCompileTime;
    using T = typename Derived::Scalar;
    Eigen::Matrix<T, 2, N> Hmat;
    Hmat.setZero();
    Hmat(0, 0) = T(1);
    Hmat(1, 1) = T(1);
    return Hmat;
  }

  template <typename Derived>
  static constexpr Eigen::Matrix<typename Derived::Scalar, 2, 1> h(
      const Eigen::MatrixBase<Derived>& x) noexcept {
    static_assert(Derived::RowsAtCompileTime >= 2 && Derived::ColsAtCompileTime == 1);
    Eigen::Matrix<typename Derived::Scalar, 2, 1> z;
    z(0) = x(0);
    z(1) = x(1);
    return z;
  }

  template <typename T>
  static void wrapInnovation(Eigen::Matrix<T, 2, 1>&) noexcept {}
};

using PixelPolicy = PixelPolicyBase<void>;
using PixelCAPolicy = PixelPolicyBase<PixelCATag>;

}  // namespace kalman
