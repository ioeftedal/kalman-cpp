#include <gtest/gtest.h>

#include <cmath>
#include <optional>

#include "helpers/alloc_counter.hpp"
#include "kalman/tracker.hpp"

// The filter runs inside the capture loop on a Jetson-class ARM target with a
// hard per-frame budget: the hot path (step / predict / roi) must not touch
// the heap.  Eigen's aligned_malloc calls std::malloc directly, so this target
// additionally interposes the malloc family via -Wl,--wrap (see
// tests/CMakeLists.txt); operator new alone would miss Eigen-internal
// allocation.

TEST(NoAlloc, HotPathIsAllocationFree) {
  kalman::Config cfg;
  cfg.pan_px_per_rad = 100.0f;
  cfg.tilt_px_per_rad = -100.0f;
  kalman::Tracker t(cfg);
  static_cast<void>(
      t.reset(kalman::Detection{400.0f, 300.0f, 5.0f}, 0.0f, 0.0f));

  alloc_counter::Scope scope;
  float checksum = 0.0f;
  for (int k = 0; k < 2000; ++k) {
    const float pan = 0.002f * static_cast<float>(k);
    const float tilt = -0.001f * static_cast<float>(k);

    std::optional<kalman::Detection> det;
    if (k % 7 != 0) {
      det = kalman::Detection{400.0f + 0.5f * static_cast<float>(k),
                              300.0f - 0.25f * static_cast<float>(k), 5.0f};
    }
    const bool accepted = t.step(pan, tilt, det);

    const kalman::Rect r = t.roi(pan, tilt, cfg.dt);
    const kalman::PredictedState p = t.predict(0.1f);
    checksum += r.cu + r.cv + r.width() + r.height() + p.position.sum() +
                p.velocity.sum() + p.covariance.sum() + accepted;
  }
  EXPECT_EQ(scope.delta(), 0u);
  EXPECT_TRUE(std::isfinite(checksum));
  EXPECT_GT(checksum, 0.0f);
}
