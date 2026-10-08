// Minimal end-to-end use of the installed kalman package: seed, step, read an
// ROI.  Exits non-zero if the filter produces nonsense.
#include <kalman/tracker.hpp>

#include <cmath>

int main() {
  kalman::Config cfg;
  cfg.pan_px_per_rad = 100.0f;
  kalman::Tracker tracker(cfg);

  if (!tracker.initialized()) {
    if (!tracker.step(0.1f, 0.0f, kalman::Detection{400.0f, 300.0f, 5.0f})) {
      return 1;
    }
  }
  for (int k = 0; k < 40; ++k) {
    static_cast<void>(
        tracker.step(0.1f, 0.0f, kalman::Detection{400.0f, 300.0f, 5.0f}));
  }
  const kalman::Rect roi = tracker.roi(0.1f, 0.0f, cfg.dt);
  const Eigen::Vector2f p = tracker.position();
  return (std::isfinite(p.x()) && std::isfinite(p.y()) &&
          roi.half_w >= cfg.roi_min_half_extent && tracker.initialized() &&
          !tracker.lost())
             ? 0
             : 1;
}
