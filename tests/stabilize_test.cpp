#include "kalman/detail/stabilize.hpp"

#include <gtest/gtest.h>

#include <cmath>

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

TEST(Stabilize, SurvivesEncoderWrapSeam) {
  Stabilizer s(100.0f, -100.0f);
  s.setReference(3.10f, 0.0f);

  // Sweep forward across the +π → -π seam.  Naively subtracting raw angles
  // would jump by 2π·gain ≈ 628 px right there, which the filter sees as the
  // target teleporting — every detection after it would be gated out.
  const float pans[] = {3.14f, -3.14f, -3.10f, -3.01f};
  float prev = s.shift(3.10f, 0.0f).x();  // zero at the reference
  for (const float pan : pans) {
    s.advance(pan, 0.0f);
    const float x = s.shift(pan, 0.0f).x();
    EXPECT_LT(std::abs(x - prev), 100.0f * 0.1f) << "pan=" << pan;
    prev = x;
  }

  // Total swept arc: 0.04 + 0.00319 (across the seam) + 0.04 + 0.09 rad.
  EXPECT_NEAR(s.shift(-3.01f, 0.0f).x(), 17.32f, 0.05f);

  // The stabilised round trip holds after the wrap too.
  const Eigen::Vector2f image(640.0f, 480.0f);
  const Eigen::Vector2f stab = s.toStabilized(image, -3.01f, 0.0f);
  EXPECT_TRUE(s.toImage(stab, -3.01f, 0.0f).isApprox(image, 1e-3f));
}

TEST(Stabilize, ConstQueryDoesNotAdvanceUnwrapHistory) {
  Stabilizer s(100.0f, -100.0f);
  s.setReference(3.10f, 0.0f);
  s.advance(3.14f, 0.0f);

  // A roi() call with a future/guessed encoder angle is a query: asking twice
  // gives the same answer, and it must not record the angle.
  const float before = s.shift(-3.10f, 0.0f).x();
  EXPECT_FLOAT_EQ(s.shift(-3.10f, 0.0f).x(), before);

  // Recording the same angle later yields the identical value...
  s.advance(-3.10f, 0.0f);
  EXPECT_FLOAT_EQ(s.shift(-3.10f, 0.0f).x(), before);

  // ...and the next real step folds exactly one step of motion, not two.
  s.advance(-3.05f, 0.0f);
  EXPECT_NEAR(s.shift(-3.05f, 0.0f).x(), before + 5.0f, 1e-3f);
}

TEST(Stabilize, NonFiniteAnglesHoldLastKnownShift) {
  Stabilizer s(100.0f, -100.0f);
  s.setReference(0.0f, 0.0f);
  s.advance(0.10f, -0.05f);
  const Eigen::Vector2f known = s.shift(0.10f, -0.05f);  // (10, 5)
  EXPECT_TRUE(known.isApprox(Eigen::Vector2f(10.0f, 5.0f), 1e-4f));

  // Queries and advances with garbage encoder angles neither move nor
  // corrupt the axis — a roi() built from a dropped telemetry frame still
  // returns a finite window.
  const float nan = std::nanf("");
  EXPECT_TRUE(s.shift(nan, nan).isApprox(known, 0.0f));
  s.advance(nan, nan);
  EXPECT_TRUE(s.shift(nan, nan).isApprox(known, 0.0f));

  // The next clean sample folds exactly one step of motion.
  s.advance(0.20f, -0.10f);
  EXPECT_TRUE(
      s.shift(0.20f, -0.10f).isApprox(Eigen::Vector2f(20.0f, 10.0f), 1e-4f));
}
