#include "kalman/detail/stabilize.hpp"

#include <gtest/gtest.h>

namespace {

using kalman::detail::Stabilizer;

}  // namespace

TEST(Stabilize, ZeroGainsAreIdentity) {
  Stabilizer s(0.0f, 0.0f);
  s.setReference(0.7f, -0.2f);
  const Eigen::Vector2f image(640.0f, 360.0f);
  EXPECT_TRUE(s.toStabilized(image, 1.3f, 0.4f).isApprox(image, 1e-6f));
  EXPECT_TRUE(s.toImage(image, 1.3f, 0.4f).isApprox(image, 1e-6f));
  EXPECT_TRUE(s.shift(1.3f, 0.4f).isApprox(Eigen::Vector2f::Zero(), 1e-6f));
}

TEST(Stabilize, ShiftIsZeroAtReference) {
  Stabilizer s(120.0f, -95.0f);
  s.setReference(0.3f, -0.15f);
  EXPECT_TRUE(s.shift(0.3f, -0.15f).isApprox(Eigen::Vector2f::Zero(), 1e-6f));
}

TEST(Stabilize, ShiftIsLinearInAngle) {
  Stabilizer s(120.0f, -95.0f);
  s.setReference(0.3f, -0.15f);
  const float dpan = 0.05f;
  const float dtilt = -0.02f;
  const Eigen::Vector2f a = s.shift(0.3f, -0.15f);
  const Eigen::Vector2f b = s.shift(0.3f + dpan, -0.15f + dtilt);
  EXPECT_TRUE(
      (b - a).isApprox(Eigen::Vector2f(120.0f * dpan, -95.0f * dtilt), 1e-5f));
}

TEST(Stabilize, ImageStabilizedRoundTrip) {
  Stabilizer s(87.5f, -103.0f);
  s.setReference(-0.4f, 0.1f);
  const Eigen::Vector2f image(123.0f, 456.0f);
  const float pan = 0.9f;
  const float tilt = -0.3f;
  const Eigen::Vector2f stab = s.toStabilized(image, pan, tilt);
  EXPECT_TRUE(s.toImage(stab, pan, tilt).isApprox(image, 1e-4f));
}
