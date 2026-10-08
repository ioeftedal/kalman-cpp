// Live-loop contract demo: what a 40 Hz capture thread feeding this library
// looks like.  The detector and the gimbal are mocked with synthetic data;
// swap `mockDetection()` for your detector's output and `pan/tilt` for your
// encoder readings.
//
// Call order per frame:
//   1. capture frame + encoder angles (pan, tilt) at exposure
//   2. run the detector on the frame (here: mocked, arriving same-frame)
//   3. tracker.step(pan, tilt, detection-or-nullopt)   -- advances one dt
//   4. tracker.roi(next_pan, next_tilt, dt)            -- search window for
//      the next frame's detection

#include <cmath>
#include <cstdio>
#include <kalman/tracker.hpp>
#include <optional>
#include <random>

namespace {

struct MockCamera {
  float pan = 0.0f;
  float tilt = 0.0f;
  void slew(float t) {  // slow scan while the drone flies
    pan = 0.15f * std::sin(0.4f * t);
    tilt = 0.05f * std::sin(0.3f * t);
  }
};

struct MockDrone {  // stabilised-frame truth: figure-8 with noise bursts
  Eigen::Vector2f pos(float t) const {
    return Eigen::Vector2f(960.0f + 300.0f * std::sin(0.9f * t),
                           540.0f + 180.0f * std::sin(1.8f * t));
  }
};

}  // namespace

int main() {
  kalman::Config cfg;
  cfg.pan_px_per_rad = 110.0f;    // hand-measured: px per rad of pan
  cfg.tilt_px_per_rad = -110.0f;  // sign depends on your encoder convention
  kalman::Tracker tracker(cfg);

  MockCamera cam;
  MockDrone drone;
  std::mt19937 rng(7);
  std::normal_distribution<float> px_noise(0.0f, 5.0f);

  const float dt = cfg.dt;
  bool seeded = false;

  std::printf("%5s %10s %10s %8s %5s %4s   %s\n", "frame", "est_u", "est_v",
              "err_px", "coast", "acc", "roi [u0 v0 u1 v1]");

  for (int frame = 0; frame < 200; ++frame) {
    const float t = static_cast<float>(frame) * dt;
    cam.slew(t);

    // Image position = stabilised truth + pan/tilt shift + detector noise.
    const Eigen::Vector2f shift(cfg.pan_px_per_rad * cam.pan,
                                cfg.tilt_px_per_rad * cam.tilt);
    const Eigen::Vector2f truth = drone.pos(t);
    const Eigen::Vector2f image = truth + shift;

    // Mock detector: misses ~8 % of frames, no detection until first frame.
    std::optional<kalman::Detection> det;
    if (frame > 0 && (frame % 13) != 5) {
      det = kalman::Detection{image.x() + px_noise(rng),
                              image.y() + px_noise(rng), 5.0f};
    }

    const bool accepted = tracker.step(cam.pan, cam.tilt, det);
    if (!tracker.initialized()) {
      continue;
    }
    seeded = true;

    // ROI for the *next* frame: lead the state by one frame period, convert
    // to that frame's image coordinates with the next encoder sample.
    const float next_pan = 0.15f * std::sin(0.4f * (t + dt));
    const float next_tilt = 0.05f * std::sin(0.3f * (t + dt));
    const kalman::Rect roi = tracker.roi(next_pan, next_tilt, dt);

    if (seeded && (frame % 20) == 0) {
      const float err = (tracker.position() - truth).norm();
      std::printf("%5d %10.1f %10.1f %8.1f %4d %5d   %.0f %.0f %.0f %.0f\n",
                  frame, tracker.position().x(), tracker.position().y(), err,
                  tracker.framesSinceUpdate(), accepted ? 1 : 0, roi.u0(),
                  roi.v0(), roi.u0() + roi.width(), roi.v0() + roi.height());
    }
  }

  std::printf(
      "\nfinal: position=(%.1f, %.1f) velocity=(%.1f, %.1f) coasting=%s\n",
      tracker.position().x(), tracker.position().y(), tracker.velocity().x(),
      tracker.velocity().y(), tracker.coasting() ? "yes" : "no");
  return 0;
}
