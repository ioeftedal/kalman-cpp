/// @file
/// @brief Data and kernels for the tracker's independent, equally tuned axes.
///
/// Both axes start with the same priors and receive the same dt, jerk noise,
/// measurement variance and joint accept/reject decision. Their covariance
/// blocks therefore remain identical; cross-axis terms remain exactly zero.
/// Store only the six unique entries of one symmetric 3x3 block, not a dense
/// 6x6 matrix. State components stay adjacent by derivative: [u,v,vu,vv,au,av].
#pragma once

#include <Eigen/Core>

#include <cmath>
#include <limits>

namespace kalman::detail {

struct AxisCovariance {
  float pp = 1.0f;
  float pv = 0.0f;
  float pa = 0.0f;
  float vv = 1.0f;
  float va = 0.0f;
  float aa = 1.0f;
};

struct FilterState {
  Eigen::Matrix<float, 6, 1> x = Eigen::Matrix<float, 6, 1>::Zero();
  AxisCovariance p;
  float last_nis = 0.0f;
};

/// Only the first diagonal of FPFᵀ + Q: enough for a search ROI.
[[nodiscard]] inline float positionVariance(const AxisCovariance& p, float h,
                                            float q) noexcept {
  const float h2 = h * h;
  const float h3 = h2 * h;
  const float h4 = h3 * h;
  return p.pp + 2.0f * h * p.pv + h2 * (p.vv + p.pa) + h3 * p.va +
         0.25f * h4 * p.aa + q * (h4 * h / 20.0f);
}

[[nodiscard]] inline AxisCovariance predictCovariance(const AxisCovariance& p,
                                                      float h,
                                                      float q) noexcept {
  const float h2 = h * h;
  const float h3 = h2 * h;
  const float h4 = h3 * h;
  return {positionVariance(p, h, q),
          p.pv + h * (p.vv + p.pa) + 1.5f * h2 * p.va + 0.5f * h3 * p.aa +
              q * (h4 / 8.0f),
          p.pa + h * p.va + 0.5f * h2 * p.aa + q * (h3 / 6.0f),
          p.vv + 2.0f * h * p.va + h2 * p.aa + q * (h3 / 3.0f),
          p.va + h * p.aa + q * (h2 / 2.0f),
          p.aa + q * h};
}

/// Materialise the public diagnostic matrix only when a caller requests it.
[[nodiscard]] inline Eigen::Matrix<float, 6, 6> expandCovariance(
    const AxisCovariance& p) noexcept {
  Eigen::Matrix<float, 6, 6> out = Eigen::Matrix<float, 6, 6>::Zero();
  for (int axis = 0; axis < 2; ++axis) {
    out(axis, axis) = p.pp;
    out(axis, axis + 2) = out(axis + 2, axis) = p.pv;
    out(axis, axis + 4) = out(axis + 4, axis) = p.pa;
    out(axis + 2, axis + 2) = p.vv;
    out(axis + 2, axis + 4) = out(axis + 4, axis + 2) = p.va;
    out(axis + 4, axis + 4) = p.aa;
  }
  return out;
}

inline void predict(FilterState& f, float h, float q) noexcept {
  f.x.head<2>() += h * f.x.segment<2>(2) + (0.5f * h * h) * f.x.tail<2>();
  f.x.segment<2>(2) += h * f.x.tail<2>();
  f.p = predictCovariance(f.p, h, q);
}

/// S = (pp + r) I₂. One scalar reciprocal replaces the 2x2 decomposition.
/// Gating is still joint (2 DOF), never an independent decision per axis.
inline bool updatePosition(FilterState& f, const Eigen::Vector2f& z, float r,
                           float gate_chi2) noexcept {
  const Eigen::Vector2f nu = z - f.x.head<2>();
  const float s = f.p.pp + r;
  if (!(s > 0.0f) || !std::isfinite(s)) {
    f.last_nis = std::numeric_limits<float>::infinity();
    return false;
  }
  const float inv_s = 1.0f / s;
  f.last_nis = nu.squaredNorm() * inv_s;
  if (!std::isfinite(f.last_nis)) {
    f.last_nis = std::numeric_limits<float>::infinity();
  }
  if (!(f.last_nis >= 0.0f) || !(f.last_nis < gate_chi2)) {
    return false;
  }

  const AxisCovariance p = f.p;
  const float kp = p.pp * inv_s;
  const float kv = p.pv * inv_s;
  const float ka = p.pa * inv_s;
  f.x.head<2>() += kp * nu;
  f.x.segment<2>(2) += kv * nu;
  f.x.tail<2>() += ka * nu;

  // Expanded Joseph form (I-KH)P(I-KH)ᵀ + rKKᵀ, not P -= KHP.
  // Keep the numerical stability of Joseph without dense matrix products.
  const float a = 1.0f - kp;
  const float vp = p.pv - kv * p.pp;
  const float ap = p.pa - ka * p.pp;
  f.p = {a * a * p.pp + r * kp * kp,
         a * vp + r * kp * kv,
         a * ap + r * kp * ka,
         p.vv - kv * p.pv - kv * vp + r * kv * kv,
         p.va - kv * p.pa - ka * vp + r * kv * ka,
         p.aa - ka * p.pa - ka * ap + r * ka * ka};
  return true;
}

}  // namespace kalman::detail
