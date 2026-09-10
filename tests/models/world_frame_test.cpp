/// @file
/// @brief World-frame pan/tilt UKF tests.

#include "kalman/models/world_frame.hpp"

#include <gtest/gtest.h>

#include "kalman/core/constants.hpp"
#include "kalman/core/filter.hpp"

using WorldCaFilter = kalman::UnscentedKalmanFilter<double, 9, 2, kalman::WorldFramePanTiltPolicy>;
using WorldCvFilter = kalman::UnscentedKalmanFilter<double, 6, 2, kalman::WorldFramePanTiltPolicy>;

// -----------------------------------------------------------------------
//  Measurement model — camera at origin (same as camera-centric)
// -----------------------------------------------------------------------

TEST(WorldFrameMeasurement, CameraAtOrigin) {
  WorldCaFilter filter;
  filter.getPolicy().setCameraPosition(0, 0, 0);
  filter.getPolicy().setOrientation(0.0, 0.0);

  // Target 100 m north (dead ahead along camera's +z for pan=tilt=0)
  Eigen::Matrix<double, 9, 1> x;
  x << 0, 100, 0, 0, 0, 0, 0, 0, 0;

  auto z = filter.getPolicy().h(x);
  EXPECT_NEAR(z(0), 0.0, 1e-10);
  EXPECT_NEAR(z(1), 0.0, 1e-10);
}

// -----------------------------------------------------------------------
//  Camera rotation invariance — rotate camera, target should shift in image
// -----------------------------------------------------------------------

TEST(WorldFrameMeasurement, CameraRotation) {
  WorldCaFilter filter;
  filter.getPolicy().setCameraPosition(0, 0, 0);

  // Target at world position (100, 0, 0) — east of camera
  Eigen::Matrix<double, 9, 1> x;
  x << 100, 0, 0, 0, 0, 0, 0, 0, 0;

  // Camera looks east (pan = +π/2 from north)
  filter.getPolicy().setOrientation(kalman::constants::pi / 2, 0.0);
  auto z = filter.getPolicy().h(x);
  // With camera looking east, target at east should be centred
  EXPECT_NEAR(z(0), 0.0, 1e-6);
}

// -----------------------------------------------------------------------
//  Camera translation
// -----------------------------------------------------------------------

TEST(WorldFrameMeasurement, CameraTranslation) {
  WorldCaFilter filter;
  filter.getPolicy().setCameraPosition(0, 10, 0);  // camera 10m north of origin
  filter.getPolicy().setOrientation(0.0, 0.0);     // looking north

  // Target also 10m north of origin → at same position as camera
  Eigen::Matrix<double, 9, 1> x;
  x << 0, 10, 0, 0, 0, 0, 0, 0, 0;

  auto z = filter.getPolicy().h(x);
  // Target is at camera origin → pan/tilt should be (0, 0)
  EXPECT_NEAR(z(0), 0.0, 1e-10);
  EXPECT_NEAR(z(1), 0.0, 1e-10);
}

// -----------------------------------------------------------------------
//  Tilt-up measurement
// -----------------------------------------------------------------------

TEST(WorldFrameMeasurement, TiltUp) {
  WorldCaFilter filter;
  filter.getPolicy().setCameraPosition(0, 0, 0);
  filter.getPolicy().setOrientation(0.0, kalman::constants::pi / 4);  // tilt up 45°

  // Target at (0, 100, 100) — 100m north, 100m up
  Eigen::Matrix<double, 9, 1> x;
  x << 0, 100, 100, 0, 0, 0, 0, 0, 0;

  auto z = filter.getPolicy().h(x);
  // With camera tilted up 45°, elevation to target is relative
  double true_elev = std::atan2(100, 100);  // 45° elevation
  EXPECT_NEAR(z(1), true_elev - kalman::constants::pi / 4, 1e-6);
}

// -----------------------------------------------------------------------
//  One-step filter check — perfect state, no innovation
// -----------------------------------------------------------------------

TEST(WorldFrameFilter, PerfectStateNoInnovation) {
  WorldCaFilter filter;
  filter.getPolicy().setCameraPosition(0, 0, 0);
  filter.getPolicy().setOrientation(0.0, 0.0);

  // True state at (0, 100, 0) — 100m north
  Eigen::Matrix<double, 9, 1> x_true;
  x_true << 0, 100, 0, 0, 0, 0, 0, 0, 0;

  filter.setState(x_true);
  filter.setCovariance(Eigen::Matrix<double, 9, 9>::Identity() * 0.01);
  filter.setProcessNoiseSpectralDensity(100.0);
  filter.setMeasurementNoiseCovariance(Eigen::Matrix2d::Identity() * 0.01);

  // Measurement at (0, 0) — target dead ahead
  filter.update(Eigen::Vector2d(0, 0));

  auto x_upd = filter.getState();
  for (int i = 0; i < 3; ++i) {
    EXPECT_NEAR(x_upd(i), x_true(i), 0.5) << "Position component " << i << " diverged";
  }
}
