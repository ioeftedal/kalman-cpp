/// @file
/// @brief 3-D world-frame pan/tilt tracking model.

#pragma once

#include <Eigen/Core>
#include <Eigen/Geometry>

#include "kalman/core/constants.hpp"
#include "kalman/core/fwd.hpp"
#include "kalman/models/pan_tilt.hpp"

namespace kalman {

/// Policy for world-frame pan/tilt bearing-only tracking.
///
/// Target state is in a fixed inertial world frame; camera extrinsics
/// (position + pan/tilt) are applied in the measurement function.
/// Set camera position once via `setCameraPosition()`, then current
/// pan/tilt before each `update()` via `getPolicy().setOrientation()`.
struct WorldFramePanTiltPolicy {
  using ModelTag = void;
  static constexpr bool isMeasurementLinear = false;

  void setCameraPosition(double x, double y, double z) noexcept {
    t_cam_ = Eigen::Vector3d(x, y, z);
  }

  void setOrientation(double pan, double tilt) noexcept {
    pan_ = pan;
    tilt_ = tilt;
  }

  const Eigen::Vector3d& getCameraPosition() const noexcept { return t_cam_; }
  double getPan() const noexcept { return pan_; }
  double getTilt() const noexcept { return tilt_; }

  template <typename Derived>
  Eigen::Matrix<typename Derived::Scalar, 2, 1> h(
      const Eigen::MatrixBase<Derived>& x) const noexcept {
    static_assert((Derived::RowsAtCompileTime == 6 || Derived::RowsAtCompileTime == 9) &&
                  Derived::ColsAtCompileTime == 1);
    using T = typename Derived::Scalar;
    using Vec3 = Eigen::Matrix<T, 3, 1>;
    using AngleAxis = Eigen::AngleAxis<T>;

    Vec3 p_rel;
    p_rel(0) = T(x(0)) - T(t_cam_(0));
    p_rel(1) = T(x(1)) - T(t_cam_(1));
    p_rel(2) = T(x(2)) - T(t_cam_(2));

    Vec3 p1 = AngleAxis(T(pan_), Vec3::UnitZ()) * p_rel;
    Vec3 p2 = AngleAxis(T(-tilt_), Vec3::UnitX()) * p1;

    Vec3 p_cam;
    p_cam(0) = p2(0);
    p_cam(1) = p2(2);
    p_cam(2) = p2(1);

    Eigen::Matrix<T, Derived::RowsAtCompileTime, 1> x_cam = x;
    x_cam(0) = p_cam(0);
    x_cam(1) = p_cam(1);
    x_cam(2) = p_cam(2);
    return kalman::h(x_cam);
  }

  template <typename T>
  static void wrapInnovation(Eigen::Matrix<T, 2, 1>& nu) noexcept {
    constexpr T two_pi = T(2) * constants::pi_v<T>;
    nu(0) = std::remainder(nu(0), two_pi);
  }

 private:
  Eigen::Vector3d t_cam_ = Eigen::Vector3d::Zero();
  double pan_ = 0.0;
  double tilt_ = 0.0;
};

}  // namespace kalman
