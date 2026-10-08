// Per-frame cost of the tracker hot path: step() + roi() + predict(), the
// three calls a 40 Hz capture loop makes every frame.
//
// Run it on the target you care about, Release build:
//
//   cmake -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build
//   ./build/benchmarks/hotpath_bench
//
// Reports the median of several batches (the min is noisy on shared/virtual
// hardware, the mean is dragged around by scheduler interruptions).
//
// The mock target is an independent ground truth in the stabilised frame,
// converted to image coordinates by recovering the gimbal shift from the
// tracker's own output — so detections are *not* derived from the estimate
// (a detection generated from the estimate feeds the estimate back into
// itself, the innovation never closes, and the filter accelerates a mirage
// off to infinity; the tracker is not at fault when a mock does that).

#include <kalman/tracker.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <optional>

namespace {

constexpr int kWarmupFrames = 2000;
constexpr int kFramesPerBatch = 10000;
constexpr int kBatches = 15;
constexpr float kDt = 1.0f / 40.0f;
constexpr float kLead = 0.1f;

/// Ground truth in the stabilised frame: a slow figure-8 with real
/// acceleration, so the CA model has something to chew on.
Eigen::Vector2f truthAt(int frame) {
  const float w = 0.01f * static_cast<float>(frame);
  return Eigen::Vector2f(400.0f + 50.0f * std::sin(w),
                         300.0f + 35.0f * std::sin(1.7f * w));
}

/// One frame of work in the order a real capture loop does it: latency lead,
/// search ROI, detector output for the truth inside it, step().  Returning
/// the ROI keeps the compiler from deleting any of it.
kalman::Rect runFrame(kalman::Tracker& t, int frame, float pan, float tilt,
                      bool have_det) {
  const kalman::PredictedState lead = t.predict(kLead);
  const kalman::Rect search = t.roi(pan, tilt, kLead);

  // roi centre = predicted position + gimbal shift (both used the same
  // horizon), so this recovers the shift the tracker applied.
  const Eigen::Vector2f shift(search.cu - lead.position.x(),
                              search.cv - lead.position.y());

  std::optional<kalman::Detection> det;
  if (have_det) {
    const Eigen::Vector2f image = truthAt(frame) + shift;
    det = kalman::Detection{image.x(), image.y(), 5.0f};
  }
  static_cast<void>(t.step(pan, tilt, det));
  return search;
}

}  // namespace

int main() {
#if defined(__GNUC__) && !defined(__OPTIMIZE__)
  std::printf(
      "warning: built without optimisation — numbers are meaningless.\n"
      "         re-run with -DCMAKE_BUILD_TYPE=Release\n\n");
#endif

  kalman::Config cfg;
  cfg.dt = kDt;  // 40 Hz capture loop, explicit rather than relying on default
  cfg.pan_px_per_rad = 110.0f;
  cfg.tilt_px_per_rad = -110.0f;
  kalman::Tracker tracker(cfg);
  static_cast<void>(
      tracker.reset(kalman::Detection{400.0f, 300.0f, 5.0f}, 0.0f, 0.0f));

  int frame = 0;
  auto frameAngles = [](int f) {
    // Slow, bounded slew — continuous across batches so the encoder unwrap
    // sees a smooth angle sequence.
    return std::pair(0.3f * std::sin(1.0e-4f * static_cast<float>(f)),
                     0.2f * std::cos(1.3e-4f * static_cast<float>(f)));
  };

  for (int k = 0; k < kWarmupFrames; ++k, ++frame) {
    const auto [pan, tilt] = frameAngles(frame);
    static_cast<void>(runFrame(tracker, frame, pan, tilt, k % 7 != 0));
  }

  double ns[kBatches];
  for (int b = 0; b < kBatches; ++b) {
    const auto t0 = std::chrono::steady_clock::now();
    for (int k = 0; k < kFramesPerBatch; ++k, ++frame) {
      const auto [pan, tilt] = frameAngles(frame);
      static_cast<void>(runFrame(tracker, frame, pan, tilt, k % 7 != 0));
    }
    const auto t1 = std::chrono::steady_clock::now();
    ns[b] = std::chrono::duration<double, std::nano>(t1 - t0).count() /
            static_cast<double>(kFramesPerBatch);
  }

  std::sort(ns, ns + kBatches);
  const double median = ns[kBatches / 2];

  // Consume the accumulated state: report the tracking error too, so a mock
  // that quietly diverges is visible in the output rather than hidden.
  const Eigen::Vector2f err = tracker.position() - truthAt(frame);
  const kalman::Rect roi = tracker.roi(0.0f, 0.0f);
  std::printf(
      "step+roi+predict per frame: median %.0f ns  (min %.0f, max %.0f)\n"
      "frames=%d x %d batches, track_err=%.1f px, roi=%.0fx%.0f\n",
      median, ns[0], ns[kBatches - 1], kFramesPerBatch, kBatches, err.norm(),
      roi.width(), roi.height());
  return std::isfinite(median) && median < 1.0e9 && err.norm() < 50.0f ? 0 : 1;
}
