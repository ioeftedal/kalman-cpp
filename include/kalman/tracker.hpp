/// @file
/// @brief Single-target pixel-space tracker for a fixed-rate pan/tilt camera.
///
/// This is the library's public API: one class, one config, one call per
/// frame.  It is built for the specific case of a stationary camera that
/// pans/tilts, runs an object detector on every frame at a fixed rate (40 Hz
/// by default), and consumes the prediction as a search ROI for the next
/// detection.
///
/// Design notes:
///  - Pixel measurements + constant-acceleration kinematics are *linear*, so
///    the core is a plain 6-D linear KF (see detail/kf.hpp) — no sigma points.
///  - Camera rotation is stripped at the boundary (detail/stabilize.hpp); the
///    state lives in a reference-stabilised frame and the filter never sees
///    gimbal motion.  Encoder angles are folded into a continuous axis every
///    frame, so the ±π wrap seam does not disturb the track.
///  - The filter clock is the measurement clock: call `step()` once per
///    captured frame, in capture order, passing `std::nullopt` when the
///    detector misses.  Detector latency never enters the state — use the
///    const `predict(horizon)` extrapolation for wall-clock-now outputs.
///  - False detections are rejected by a chi-square innovation gate and the
///    state is left untouched; the track coasts on prediction alone until the
///    detector recovers.
///  - A track whose detector stays blind for `Config::max_coast_frames`
///    consecutive frames is declared `lost()`; from then on `step()` only
///    re-seeds (the coasted state is no longer worth trusting) and the caller
///    should widen its search until a detection arrives.
///
/// Robustness: `Config` is copied and *sanitised* at construction (non-finite
/// or out-of-range fields fall back to their documented defaults — see
/// `Config`), `reset()` refuses a non-finite seed, and every runtime float
/// (`dt`, `horizon_s`, `k_sigma`, encoder angles, detection coordinates) is
/// validated before it can reach the state.  Invalid input degrades to "no
/// information applied", never to a corrupted state.
///
/// Threading: an instance is not thread-safe; use one per detection thread.

#pragma once

#include <Eigen/Core>
#include <algorithm>
#include <cmath>
#include <limits>
#include <optional>

#include "kalman/detail/ca_model.hpp"
#include "kalman/detail/kf.hpp"
#include "kalman/detail/stabilize.hpp"

namespace kalman {

/// One detector output: centre position in image pixels.
struct Detection {
  float u = 0.0f;
  float v = 0.0f;
  /// 1-sigma position uncertainty in pixels (e.g. from the detector's
  /// confidence).  Values `<= 0` or non-finite fall back to
  /// `Config::detection_sigma`.
  float sigma = 0.0f;
};

/// Filter tuning.  Defaults target a 40 Hz camera and pixel units; the
/// process noise `q_jerk` is the knob that matters — see README.
///
/// The `Tracker` constructor copies this struct and replaces any field that is
/// non-finite or outside its documented range with the default shown here, so
/// a bad telemetry/config path can never poison the filter.  `Tracker::config()`
/// reports the sanitised values actually in use.  The config is immutable
/// after construction — build a new `Tracker` to retune.
struct Config {
  /// Nominal frame period, seconds.  40 Hz camera.  Must be finite and > 0.
  float dt = 1.0f / 40.0f;
  /// White-jerk spectral density, px²/s⁵.  Set it so the acceleration can
  /// change by your target's maximum apparent pull (in pixels/s²) over ~0.5 s:
  /// `q ≈ (max_accel_px_per_s²)² / 0.5`.  Must be finite and >= 0.
  float q_jerk = 2.0e5f;
  /// Assumed detection sigma when `Detection::sigma` is not usable, px.
  /// Must be finite and > 0.
  float detection_sigma = 5.0f;
  /// Innovation gate: chi-square quantile for the 2-DOF NIS (9.21 = 99 %).
  /// Rejected measurements leave the state untouched.  Must be finite and > 0.
  float gate_chi2 = 9.21f;
  /// Initial 1-sigma uncertainties for `reset()`, finite and >= 0.
  float init_pos_sigma = 30.0f;    // px
  float init_vel_sigma = 500.0f;   // px/s
  float init_acc_sigma = 1500.0f;  // px/s²
  /// ROI half-extent = `max(roi_k_sigma · σ, roi_min_half_extent)` per axis.
  float roi_k_sigma = 3.0f;  // finite and > 0
  float roi_min_half_extent =
      40.0f;  // px — at least the target's box size; finite and >= 0
  /// Pan/tilt rotation→pixel gains, px/rad, sign included (0 = no
  /// compensation; non-finite falls back to 0).  Measure: sweep the encoder
  /// across a static scene and divide the pixel displacement by the angle
  /// swept.
  float pan_px_per_rad = 0.0f;
  float tilt_px_per_rad = 0.0f;
  /// Consecutive frames without an accepted detection after which the track
  /// is declared `lost()` and must be re-seeded.  0 (or negative) disables
  /// loss detection, letting a track coast indefinitely.
  int max_coast_frames = 120;  // 3 s at 40 Hz
};

/// Image-space search window: centre + half extents.
struct Rect {
  float cu = 0.0f;
  float cv = 0.0f;
  float half_w = 0.0f;
  float half_h = 0.0f;

  [[nodiscard]] float width() const noexcept { return 2.0f * half_w; }
  [[nodiscard]] float height() const noexcept { return 2.0f * half_h; }
  [[nodiscard]] float u0() const noexcept { return cu - half_w; }
  [[nodiscard]] float v0() const noexcept { return cv - half_h; }
};

/// State estimate at `t + horizon`, in the stabilised frame.
struct PredictedState {
  Eigen::Vector2f position = Eigen::Vector2f::Zero();
  Eigen::Vector2f velocity = Eigen::Vector2f::Zero();
  Eigen::Matrix<float, 6, 6> covariance =
      Eigen::Matrix<float, 6, 6>::Identity();
};

class Tracker {
 public:
  static constexpr int kStateDim = 6;
  static constexpr int kMeasDim = 2;
  using Vector6 = Eigen::Matrix<float, 6, 1>;
  using Matrix6 = Eigen::Matrix<float, 6, 6>;
  using Filter = detail::LinearFilter<kStateDim, kMeasDim>;

  explicit Tracker(const Config& cfg = Config())
      : cfg_(sanitize(cfg)),
        stabilizer_(cfg_.pan_px_per_rad, cfg_.tilt_px_per_rad) {}

  /// Seed the track from the first detection.  The supplied encoder angles
  /// become the stabilised-frame reference.  Subsequent frames use `step()`.
  ///
  /// @return false — leaving the tracker untouched — when the seed or the
  /// encoder angles are not finite (a poisoned seed could never recover).
  [[nodiscard]] bool reset(const Detection& first, float pan_rad,
                           float tilt_rad) {
    if (!allFinite(first.u) || !allFinite(first.v) || !allFinite(pan_rad) ||
        !allFinite(tilt_rad)) {
      return false;
    }
    stabilizer_.setReference(pan_rad, tilt_rad);
    const Eigen::Vector2f z = stabilizer_.toStabilized(
        Eigen::Vector2f(first.u, first.v), pan_rad, tilt_rad);

    Vector6 x = Vector6::Zero();
    x.head<2>() = z;
    Matrix6 P = Matrix6::Zero();
    P(0, 0) = P(1, 1) = sq(cfg_.init_pos_sigma);
    P(2, 2) = P(3, 3) = sq(cfg_.init_vel_sigma);
    P(4, 4) = P(5, 5) = sq(cfg_.init_acc_sigma);
    filter_.setState(x);
    filter_.setCovariance(P);

    frames_since_update_ = 0;
    initialized_ = true;
    return true;
  }

  /// Advance one frame and optionally apply this frame's detection.
  ///
  /// @param pan_rad  encoder angle at frame exposure (stabilisation input).
  /// @param tilt_rad encoder angle at frame exposure.
  /// @param det      this frame's detection, or `std::nullopt` on a miss.
  /// @param dt       frame period override in seconds; non-finite or `<= 0`
  ///                 uses `Config::dt`.
  /// @return true when a detection was accepted (or the track was seeded).
  ///
  /// The first call with a detection initialises the track (equivalent to
  /// `reset`) without a predict step, so the one-call-per-frame discipline
  /// holds from frame one.  A rejected (gated) detection advances the clock
  /// and returns false.
  ///
  /// A frame whose encoder angles or detection are non-finite is dropped: the
  /// clock still advances, but no measurement is applied (a detection cannot
  /// be mapped into the stabilised frame without valid encoder angles).
  ///
  /// Once `lost()` is true the track is frozen: only a detection re-seeds it,
  /// anything else returns false.
  [[nodiscard]] bool step(float pan_rad, float tilt_rad,
                          std::optional<Detection> det = std::nullopt,
                          float dt = -1.0f) {
    if (!initialized_ || lost()) {
      if (!det) {
        return false;
      }
      return reset(*det, pan_rad, tilt_rad);
    }

    // Fold this frame's encoder angles into the continuous axis whether or not
    // the detector produced anything — the unwrap history needs every sample
    // to stay continuous across the ±π seam.  Invalid angles skip the fold
    // and poison nothing; the measurement below is then dropped.
    const bool encoder_ok = allFinite(pan_rad) && allFinite(tilt_rad);
    if (encoder_ok) {
      stabilizer_.advance(pan_rad, tilt_rad);
    }

    const float h = framePeriod(dt);
    filter_.predict(detail::caTransition(h), detail::caNoise(h, cfg_.q_jerk));

    if (!det || !encoder_ok) {
      bumpCoast();
      return false;
    }

    const float sigma = (det->sigma > 0.0f && allFinite(det->sigma))
                            ? det->sigma
                            : cfg_.detection_sigma;
    const Eigen::Vector2f z = stabilizer_.toStabilized(
        Eigen::Vector2f(det->u, det->v), pan_rad, tilt_rad);

    const bool accepted =
        filter_.updatePosition(z, sigma * sigma, cfg_.gate_chi2);
    if (accepted) {
      frames_since_update_ = 0;
    } else {
      bumpCoast();
    }
    return accepted;
  }

  /// Smoothed position in the stabilised frame, px.
  [[nodiscard]] Eigen::Vector2f position() const noexcept {
    return filter_.state().head<2>();
  }

  /// Smoothed velocity in the stabilised frame, px/s.
  [[nodiscard]] Eigen::Vector2f velocity() const noexcept {
    return filter_.state().segment<2>(2);
  }

  /// Full 6-D state `[u, v, vu, vv, au, av]` (stabilised frame).
  [[nodiscard]] const Vector6& state() const noexcept {
    return filter_.state();
  }
  [[nodiscard]] const Matrix6& covariance() const noexcept {
    return filter_.covariance();
  }

  /// Const extrapolation of the state to `t + horizon_s` — no mutation, so it
  /// is safe to call from anywhere (e.g. latency compensation, ROI lead).
  /// A non-finite or negative horizon is treated as 0.
  [[nodiscard]] PredictedState predict(float horizon_s) const {
    const float horizon =
        (allFinite(horizon_s) && horizon_s >= 0.0f) ? horizon_s : 0.0f;
    const Matrix6 F = detail::caTransition(horizon);
    const Matrix6 Q = detail::caNoise(horizon, cfg_.q_jerk);

    PredictedState out;
    const Vector6 x = F * filter_.state();
    out.position = x.head<2>();
    out.velocity = x.segment<2>(2);
    out.covariance = F * filter_.covariance() * F.transpose() + Q;
    out.covariance = 0.5f * (out.covariance + out.covariance.transpose());
    return out;
  }

  /// Image-space search ROI for a frame whose encoder reads
  /// (`pan_rad`, `tilt_rad`), leading the state by `horizon_s`.
  ///
  /// Half-extents are `max(roi_k_sigma · σ, roi_min_half_extent)` per axis;
  /// σ grows with the coast duration and with `horizon_s`, so a coasting track
  /// automatically widens the detector's search window.  A non-finite or
  /// non-positive `k_sigma` falls back to `Config::roi_k_sigma`.
  [[nodiscard]] Rect roi(float pan_rad, float tilt_rad, float horizon_s = 0.0f,
                         float k_sigma = -1.0f) const {
    const float k =
        (allFinite(k_sigma) && k_sigma > 0.0f) ? k_sigma : cfg_.roi_k_sigma;
    const PredictedState p = predict(horizon_s);
    const Eigen::Vector2f c =
        stabilizer_.toImage(p.position, pan_rad, tilt_rad);

    Rect r;
    r.cu = c.x();
    r.cv = c.y();
    r.half_w = std::max(k * std::sqrt(std::max(p.covariance(0, 0), 0.0f)),
                        cfg_.roi_min_half_extent);
    r.half_h = std::max(k * std::sqrt(std::max(p.covariance(1, 1), 0.0f)),
                        cfg_.roi_min_half_extent);
    return r;
  }

  [[nodiscard]] bool initialized() const noexcept { return initialized_; }
  /// True while the track runs on prediction alone (no accepted detection
  /// since the last `step`).
  [[nodiscard]] bool coasting() const noexcept {
    return initialized_ && frames_since_update_ > 0;
  }
  /// True once the detector has been blind for `Config::max_coast_frames`
  /// consecutive frames.  The estimate is no longer trustworthy: widen the
  /// search (or go full-frame) until the next detection re-seeds the track.
  /// Always false when `max_coast_frames <= 0`.
  [[nodiscard]] bool lost() const noexcept {
    return initialized_ && cfg_.max_coast_frames > 0 &&
           frames_since_update_ >= cfg_.max_coast_frames;
  }
  [[nodiscard]] int framesSinceUpdate() const noexcept {
    return frames_since_update_;
  }
  /// NIS of the most recent update attempt (accepted or gated out); 0 until
  /// the first update after a `reset()`.
  [[nodiscard]] float lastNis() const noexcept { return filter_.lastNis(); }
  [[nodiscard]] const Config& config() const noexcept { return cfg_; }

 private:
  static constexpr float sq(float x) noexcept { return x * x; }

  static bool allFinite(float x) noexcept { return std::isfinite(x); }

  /// Replace unusable fields with the documented defaults.
  static Config sanitize(const Config& in) noexcept {
    Config c;  // defaults
    if (allFinite(in.dt) && in.dt > 0.0f)
      c.dt = in.dt;
    if (allFinite(in.q_jerk) && in.q_jerk >= 0.0f)
      c.q_jerk = in.q_jerk;
    if (allFinite(in.detection_sigma) && in.detection_sigma > 0.0f) {
      c.detection_sigma = in.detection_sigma;
    }
    if (allFinite(in.gate_chi2) && in.gate_chi2 > 0.0f)
      c.gate_chi2 = in.gate_chi2;
    if (allFinite(in.init_pos_sigma) && in.init_pos_sigma >= 0.0f) {
      c.init_pos_sigma = in.init_pos_sigma;
    }
    if (allFinite(in.init_vel_sigma) && in.init_vel_sigma >= 0.0f) {
      c.init_vel_sigma = in.init_vel_sigma;
    }
    if (allFinite(in.init_acc_sigma) && in.init_acc_sigma >= 0.0f) {
      c.init_acc_sigma = in.init_acc_sigma;
    }
    if (allFinite(in.roi_k_sigma) && in.roi_k_sigma > 0.0f) {
      c.roi_k_sigma = in.roi_k_sigma;
    }
    if (allFinite(in.roi_min_half_extent) && in.roi_min_half_extent >= 0.0f) {
      c.roi_min_half_extent = in.roi_min_half_extent;
    }
    if (allFinite(in.pan_px_per_rad))
      c.pan_px_per_rad = in.pan_px_per_rad;
    if (allFinite(in.tilt_px_per_rad))
      c.tilt_px_per_rad = in.tilt_px_per_rad;
    c.max_coast_frames = in.max_coast_frames;
    return c;
  }

  [[nodiscard]] float framePeriod(float dt) const noexcept {
    // cfg_.dt was sanitised at construction, so it is a usable fallback.
    return (allFinite(dt) && dt > 0.0f) ? dt : cfg_.dt;
  }

  void bumpCoast() noexcept {
    if (frames_since_update_ < std::numeric_limits<int>::max()) {
      ++frames_since_update_;
    }
  }

  Config cfg_;
  Filter filter_;
  detail::Stabilizer stabilizer_;
  int frames_since_update_ = 0;
  bool initialized_ = false;
};

}  // namespace kalman
