/// @file
/// @brief 1-D thermal constant-velocity model — example of adding a new model.
///
/// State: [temperature, dT/dt]  (N=2, M=1)
/// Measurement: direct temperature reading (linear)
///
/// This file demonstrates the minimal boilerplate needed to add a new
/// Kalman filter variant: define a ModelTag, specialise ProcessModel,
/// and write a Policy struct.

#pragma once

#include <Eigen/Core>

#include "kalman/core/fwd.hpp"

namespace kalman {

/// Tag that disambiguates the 2-state thermal process model from other
/// potential N=2 models.
struct ThermalTag {};

/// Process model specialisation for 1-D thermal CV (N=2).
///
/// State: [T, dT/dt]
///   F = [1  dt; 0  1]
///   Q = q_c · [dt³/3  dt²/2;  dt²/2  dt]
template <typename T>
struct ProcessModel<T, 2, ThermalTag> {
  static constexpr Eigen::Matrix<T, 2, 2> F(T dt) noexcept {
    Eigen::Matrix<T, 2, 2> Fmat;
    Fmat << 1, dt, 0, 1;
    return Fmat;
  }

  static constexpr Eigen::Matrix<T, 2, 2> Q(T dt, T q_c) noexcept {
    T dt2 = dt * dt / 2;
    T dt3 = dt * dt * dt / 3;
    Eigen::Matrix<T, 2, 2> Qmat;
    Qmat << dt3, dt2, dt2, dt;
    return q_c * Qmat;
  }
};

/// Policy for 1-D thermal CV tracking.
///
/// Measurement: direct temperature reading.
///   H = [1, 0]
///   h(x) = [x(0)]
struct ThermalPolicy {
  using ModelTag = ThermalTag;
  static constexpr bool isMeasurementLinear = true;

  template <typename Derived>
  static constexpr auto H(const Eigen::MatrixBase<Derived>&) noexcept {
    Eigen::Matrix<typename Derived::Scalar, 1, 2> Hmat;
    Hmat << 1, 0;
    return Hmat;
  }

  template <typename Derived>
  static constexpr Eigen::Matrix<typename Derived::Scalar, 1, 1> h(
      const Eigen::MatrixBase<Derived>& x) noexcept {
    Eigen::Matrix<typename Derived::Scalar, 1, 1> z;
    z(0) = x(0);
    return z;
  }

  static void wrapInnovation(auto&) noexcept {}
};

}  // namespace kalman
