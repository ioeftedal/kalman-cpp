// Identical precomputed inputs for every batch; generation is not timed.
// Consume complete outputs, including covariance, so dead-code elimination
// cannot turn predict()/roi() into a cheaper, unrepresentative workload.
#include <kalman/tracker.hpp>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <optional>

namespace {
constexpr int kWarmup = 2000;
constexpr int kFrames = 10000;
constexpr int kBatches = 15;
constexpr float kLead = 0.1f;

struct Sample {
  float pan;
  float tilt;
  kalman::Detection detection;
  Eigen::Vector2f truth;
};

bool benchmark(const char* name, bool mixed, bool coast) {
  kalman::Config cfg;
  cfg.pan_px_per_rad = 110.0f;
  cfg.tilt_px_per_rad = -110.0f;
  cfg.max_coast_frames = 0;
  std::array<Sample, kWarmup + kFrames> samples;
  for (int i = 0; i < kWarmup + kFrames; ++i) {
    const float f = static_cast<float>(i);
    auto& s = samples[i];
    s.pan = 0.3f * std::sin(1.0e-4f * f);
    s.tilt = 0.2f * std::sin(1.3e-4f * f);
    s.truth = {400.0f + 50.0f * std::sin(0.01f * f),
               300.0f + 35.0f * std::sin(0.017f * f)};
    s.detection = {s.truth.x() + cfg.pan_px_per_rad * s.pan,
                   s.truth.y() + cfg.tilt_px_per_rad * s.tilt, 5.0f};
    if (mixed && i % 31 == 0) {
      s.detection.u += 1000.0f;
    }
  }

  std::array<double, kBatches> ns;
  int accepted = 0;
  float error = 0.0f;
  for (int b = 0; b < kBatches; ++b) {
    kalman::Tracker tracker(cfg);
    static_cast<void>(tracker.reset({400.0f, 300.0f, 5.0f}, 0.0f, 0.0f));
    auto run = [&](int i) {
      const auto& s = samples[i];
      const auto lead = tracker.predict(kLead);
      const auto search = tracker.roi(s.pan, s.tilt, kLead);
      const std::optional<kalman::Detection> det =
          (coast || (mixed && i % 7 == 0))
              ? std::nullopt
              : std::optional<kalman::Detection>(s.detection);
      const bool ok = tracker.step(s.pan, s.tilt, det);
      // GCC/Clang compiler barrier: make every byte observable without
      // introducing a volatile store or checksum loop into the timed path.
      asm volatile("" : : "g"(&lead), "g"(&search) : "memory");
      return ok;
    };
    for (int i = 0; i < kWarmup; ++i) {
      static_cast<void>(run(i));
    }
    accepted = 0;
    const auto start = std::chrono::steady_clock::now();
    for (int i = kWarmup; i < kWarmup + kFrames; ++i) {
      accepted += run(i);
    }
    ns[b] = std::chrono::duration<double, std::nano>(
                std::chrono::steady_clock::now() - start)
                .count() /
            kFrames;
    error = (tracker.position() - samples.back().truth).norm();
  }
  std::sort(ns.begin(), ns.end());
  std::printf(
      "%-8s step+roi+predict: median %.0f ns (min %.0f, max %.0f), "
      "accepted=%d/%d, error=%.3f px\n",
      name, ns[kBatches / 2], ns.front(), ns.back(), accepted, kFrames, error);
  return std::isfinite(error) &&
         (coast ? accepted == 0 : accepted > kFrames / 2 && error < 5.0f);
}
}  // namespace

int main() {
#if defined(__GNUC__) && !defined(__OPTIMIZE__)
  std::printf("warning: use -DCMAKE_BUILD_TYPE=Release for timing\n");
#endif
  std::printf("Tracker: %zu bytes; %d frames x %d batches\n",
              sizeof(kalman::Tracker), kFrames, kBatches);
  const bool tracked = benchmark("tracked", false, false);
  const bool mixed = benchmark("mixed", true, false);
  const bool coast = benchmark("coast", false, true);
  return tracked && mixed && coast ? 0 : 1;
}
