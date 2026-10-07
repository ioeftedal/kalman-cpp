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
///    gimbal motion.
///  - The filter clock is the measurement clock: call `step()` once per
///    captured frame, in capture order, passing `std::nullopt` when the
///    detector misses.  Detector latency never enters the state — use the
///    const `predict(horizon)` extrapolation for wall-clock-now outputs.
///  - False detections are rejected by a chi-square innovation gate and the
///    state is left untouched; the track coasts on prediction alone until the
///    detector recovers.
///
/// Threading: an instance is not thread-safe; use one per detection thread.

#pragma once

#include <Eigen/Core>
#include <algorithm>
#include <cmath>
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
  /// confidence).  Values `<= 0` fall back to `Config::detection_sigma`.
  float sigma = 0.0f;
};

/// Filter tuning.  Defaults target a 40 Hz camera and pixel units; the
/// process noise `q_jerk` is the knob that matters — see README.
struct Config {
  /// Nominal frame period, seconds.  40 Hz camera.
  float dt = 1.0f / 40.0f;
  /// White-jerk spectral density, px²/s⁵.  Set it so the acceleration can
  /// change by your target's maximum apparent pull (in pixels) over ~0.5 s:
  /// `q ≈ (max_accel_px)² / 0.5`.
  float q_jerk = 2.0e5f;
  /// Assumed detection sigma when `Detection::sigma` is not supplied, px.
  float detection_sigma = 5.0f;
  /// Innovation gate: chi-square quantile for the 2-DOF NIS (9.21 = 99 %).
  /// Rejected measurements leave the state untouched.
  float gate_chi2 = 9.21f;
  /// Initial 1-sigma uncertainties for `reset()`.
  float init_pos_sigma = 30.0f;    // px
  float init_vel_sigma = 500.0f;   // px/s
  float init_acc_sigma = 1500.0f;  // px/s²
  /// ROI half-extent = `max(roi_k_sigma · σ, roi_min_half_extent)` per axis.
  float roi_k_sigma = 3.0f;
  float roi_min_half_extent = 40.0f;  // px — at least the target's box size
  /// Pan/tilt rotation→pixel gains, px/rad, sign included (0 = no
  /// compensation).  Measure: sweep the encoder across a static scene and
  /// divide the pixel displacement by the angle swept.
  float pan_px_per_rad = 0.0f;
  float tilt_px_per_rad = 0.0f;
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
      : cfg_(cfg), stabilizer_(cfg.pan_px_per_rad, cfg.tilt_px_per_rad) {}

  /// Seed the track from the first detection.  The supplied encoder angles
  /// become the stabilised-frame reference.  Subsequent frames use `step()`.
  void reset(const Detection& first, float pan_rad, float tilt_rad) {
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
  }

  /// Advance one frame and optionally apply this frame's detection.
  ///
  /// @param pan_rad  encoder angle at frame exposure (stabilisation input).
  /// @param tilt_rad encoder angle at frame exposure.
  /// @param det      this frame's detection, or `std::nullopt` on a miss.
  /// @param dt       frame period override in seconds; `<= 0` uses `Config::dt`.
  /// @return true when a detection was accepted (or the track was seeded).
  ///
  /// The first call with a detection initialises the track (equivalent to
  /// `reset`) without a predict step, so the one-call-per-frame discipline
  /// holds from frame one.  A rejected (gated) detection advances the clock
  /// and returns false.
  bool step(float pan_rad, float tilt_rad,
            std::optional<Detection> det = std::nullopt, float dt = -1.0f) {
    if (!initialized_) {
      if (!det) {
        return false;
      }
      reset(*det, pan_rad, tilt_rad);
      return true;
    }

    const float h = dt > 0.0f ? dt : cfg_.dt;
    filter_.predict(detail::caTransition(h), detail::caNoise(h, cfg_.q_jerk));

    if (!det) {
      ++frames_since_update_;
      return false;
    }

    const float sigma = det->sigma > 0.0f ? det->sigma : cfg_.detection_sigma;
    const Eigen::Vector2f z = stabilizer_.toStabilized(
        Eigen::Vector2f(det->u, det->v), pan_rad, tilt_rad);

    Eigen::Matrix<float, kMeasDim, kMeasDim> R =
        Eigen::Matrix<float, kMeasDim, kMeasDim>::Identity() * (sigma * sigma);
    Eigen::Matrix<float, kMeasDim, kStateDim> H =
        Eigen::Matrix<float, kMeasDim, kStateDim>::Zero();
    H(0, 0) = 1.0f;
    H(1, 1) = 1.0f;

    const bool accepted = filter_.update(z, R, H, cfg_.gate_chi2);
    if (accepted) {
      frames_since_update_ = 0;
    } else {
      ++frames_since_update_;
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
  [[nodiscard]] PredictedState predict(float horizon_s) const {
    const Matrix6 F = detail::caTransition(horizon_s);
    const Matrix6 Q = detail::caNoise(horizon_s, cfg_.q_jerk);

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
  /// automatically widens the detector's search window.
  [[nodiscard]] Rect roi(float pan_rad, float tilt_rad, float horizon_s = 0.0f,
                         float k_sigma = -1.0f) const {
    const float k = k_sigma > 0.0f ? k_sigma : cfg_.roi_k_sigma;
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
  [[nodiscard]] int framesSinceUpdate() const noexcept {
    return frames_since_update_;
  }
  /// NIS of the most recent update attempt (accepted or gated out).
  [[nodiscard]] float lastNis() const noexcept { return filter_.lastNis(); }
  [[nodiscard]] const Config& config() const noexcept { return cfg_; }

 private:
  static constexpr float sq(float x) noexcept { return x * x; }

  Config cfg_;
  Filter filter_;
  detail::Stabilizer stabilizer_;
  int frames_since_update_ = 0;
  bool initialized_ = false;
};

}  // namespace kalman
