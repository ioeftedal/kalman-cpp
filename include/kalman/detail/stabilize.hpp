/// @file
/// @brief Pan/tilt image-motion compensation via a reference-stabilised frame.
///
/// The camera only rotates (it is otherwise fixed), and pure rotation shifts
/// image content by an amount that is independent of scene depth — so with
/// exposure-synced encoder angles the camera's contribution to apparent target
/// motion is a *known deterministic input*, not something to estimate.  The
/// tracker therefore lives in a reference ("stabilised") pixel frame fixed at
/// the orientation recorded by `setReference()`; the shift below is subtracted
/// from incoming detections and added back when producing image-space output.
/// The filter never sees gimbal motion, so one tuning set works at any slew
/// rate.
///
/// **Encoder wrapping.**  PTZ encoders report absolute angles in [-π, π] and
/// wrap when the gimbal crosses the seam.  Subtracting raw angles would then
/// jump by 2π · gain (≈ 600 px for a 100 px/rad camera), which looks to the
/// filter like the target teleported — every subsequent detection gets gated
/// out and the track dies.  Angles are therefore folded into a *continuous*
/// axis: `advance()` accumulates the shortest-arc delta each frame, and the
/// reference is the origin of that axis.  Queries (`shift`, `toStabilized`,
/// `toImage`) unwrap against the current axis state without mutating it, so a
/// `roi()` call with a future/guessed encoder angle cannot corrupt the
/// history.  Drift between two `advance()` calls must stay within ±π —
/// trivially true when it is called once per captured frame.
///
/// Without calibration the rotation→pixel mapping is a single measured gain
/// per axis (sign included): point the camera at a static target, sweep the
/// encoder by Δ, read the pixel displacement Δu → `pan_px_per_rad = Δu/Δpan`.
/// That is the small-angle model `shift = gain · (angle − reference)`; an
/// exact ray projection can replace it later behind the same interface once
/// intrinsics exist.

#pragma once

#include <Eigen/Core>

#include <cmath>

namespace kalman::detail {

class Stabilizer {
 public:
  Stabilizer(float pan_px_per_rad, float tilt_px_per_rad) noexcept
      : pan_gain_(pan_px_per_rad), tilt_gain_(tilt_px_per_rad) {}

  /// (Re-)anchor the stabilised frame: this orientation becomes the origin of
  /// the continuous angle axis.  Called by `Tracker::reset()` at seed time.
  void setReference(float pan_rad, float tilt_rad) noexcept {
    raw_pan_ = pan_rad;
    raw_tilt_ = tilt_rad;
    un_pan_ = 0.0f;
    un_tilt_ = 0.0f;
  }

  /// Fold this frame's raw encoder angles into the continuous axis.  Call once
  /// per captured frame with exposure-synced angles, detection or not — the
  /// unwrap history needs every sample to stay continuous across the seam.
  /// Non-finite angles carry no information and are ignored (the axis holds
  /// its last known value rather than being poisoned by a NaN).
  void advance(float pan_rad, float tilt_rad) noexcept {
    if (std::isfinite(pan_rad)) {
      un_pan_ += wrapPi(pan_rad - raw_pan_);
      raw_pan_ = pan_rad;
    }
    if (std::isfinite(tilt_rad)) {
      un_tilt_ += wrapPi(tilt_rad - raw_tilt_);
      raw_tilt_ = tilt_rad;
    }
  }

  /// Pixel displacement of image content relative to the reference frame.
  /// Const: unwraps against the current axis state without advancing it.
  [[nodiscard]] Eigen::Vector2f shift(float pan_rad,
                                      float tilt_rad) const noexcept {
    return Eigen::Vector2f(
        pan_gain_ * (unwrapped(pan_rad, raw_pan_, un_pan_)),
        tilt_gain_ * (unwrapped(tilt_rad, raw_tilt_, un_tilt_)));
  }

  /// Image-space point → stabilised frame.
  [[nodiscard]] Eigen::Vector2f toStabilized(const Eigen::Vector2f& image,
                                             float pan_rad,
                                             float tilt_rad) const noexcept {
    return image - shift(pan_rad, tilt_rad);
  }

  /// Stabilised point → image-space coordinates of a frame at the given
  /// encoder angles.
  [[nodiscard]] Eigen::Vector2f toImage(const Eigen::Vector2f& stabilized,
                                        float pan_rad,
                                        float tilt_rad) const noexcept {
    return stabilized + shift(pan_rad, tilt_rad);
  }

 private:
  /// Shortest signed arc from `d` onto (-π, π].
  static float wrapPi(float d) noexcept {
    return std::remainder(d, 6.28318530717958647692f);
  }

  /// Virtual unwrap: where `raw` sits on the continuous axis anchored at the
  /// last `advance()`'d sample, without recording it.  A non-finite `raw`
  /// carries no information: hold the axis where it is (so a garbage encoder
  /// sample in a `roi()` query cannot produce a NaN window).
  static float unwrapped(float raw, float raw_prev, float un_prev) noexcept {
    if (!std::isfinite(raw)) {
      return un_prev;
    }
    return un_prev + wrapPi(raw - raw_prev);
  }

  float pan_gain_;
  float tilt_gain_;
  float raw_pan_ = 0.0f;  // last advance()'d raw angles
  float raw_tilt_ = 0.0f;
  float un_pan_ = 0.0f;  // continuous axis value at that sample
  float un_tilt_ = 0.0f;
};

}  // namespace kalman::detail
