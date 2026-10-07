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
/// Without calibration the rotation→pixel mapping is a single measured gain
/// per axis (sign included): point the camera at a static target, sweep the
/// encoder by Δ, read the pixel displacement Δu → `pan_px_per_rad = Δu/Δpan`.
/// That is the small-angle model `shift = gain · (angle − reference)`; an
/// exact ray projection can replace it later behind the same interface once
/// intrinsics exist.

#pragma once

#include <Eigen/Core>

namespace kalman::detail {

class Stabilizer {
 public:
  Stabilizer(float pan_px_per_rad, float tilt_px_per_rad) noexcept
      : pan_gain_(pan_px_per_rad), tilt_gain_(tilt_px_per_rad) {}

  /// Record the reference encoder orientation (first frame / `reset()`).
  void setReference(float pan_rad, float tilt_rad) noexcept {
    pan_ref_ = pan_rad;
    tilt_ref_ = tilt_rad;
  }

  /// Pixel displacement of image content relative to the reference frame.
  [[nodiscard]] Eigen::Vector2f shift(float pan_rad,
                                      float tilt_rad) const noexcept {
    return Eigen::Vector2f(pan_gain_ * (pan_rad - pan_ref_),
                           tilt_gain_ * (tilt_rad - tilt_ref_));
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
  float pan_gain_;
  float tilt_gain_;
  float pan_ref_ = 0.0f;
  float tilt_ref_ = 0.0f;
};

}  // namespace kalman::detail
