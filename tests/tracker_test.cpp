#include "kalman/tracker.hpp"

#include <gtest/gtest.h>

#include <Eigen/Cholesky>

#include <cmath>
#include <limits>
#include <random>
#include <vector>

#include "kalman/detail/ca_model.hpp"

namespace {

using kalman::Config;
using kalman::Detection;
using kalman::Tracker;

constexpr float kDt = 1.0f / 40.0f;

/// Deterministic normal noise source shared by truth generator and detector.
class Noise {
 public:
  explicit Noise(unsigned seed) : gen_(seed) {}
  float operator()() { return dist_(gen_); }

 private:
  std::mt19937 gen_;
  std::normal_distribution<float> dist_{0.0f, 1.0f};
};

struct Truth {
  Eigen::Vector2f pos = Eigen::Vector2f::Zero();
  Eigen::Vector2f vel = Eigen::Vector2f::Zero();
  Eigen::Vector2f acc = Eigen::Vector2f::Zero();

  void advance(float dt) {
    pos += vel * dt + acc * (0.5f * dt * dt);
    vel += acc * dt;
  }
};

Detection noisyDetection(const Truth& t, float sigma, Noise& n) {
  return Detection{t.pos.x() + sigma * n(), t.pos.y() + sigma * n(), sigma};
}

}  // namespace

TEST(Tracker, UninitialisedStepSeedsFromFirstDetection) {
  Tracker t;
  EXPECT_FALSE(t.initialized());

  // A miss before the first detection cannot seed the track.
  EXPECT_FALSE(t.step(0.0f, 0.0f));
  EXPECT_FALSE(t.initialized());

  const Detection d{400.0f, 300.0f, 5.0f};
  EXPECT_TRUE(t.step(0.1f, -0.05f, d));
  EXPECT_TRUE(t.initialized());
  EXPECT_FALSE(t.coasting());
  EXPECT_TRUE(t.position().isApprox(Eigen::Vector2f(400.0f, 300.0f), 1e-3f));
  EXPECT_TRUE(t.velocity().isZero());
}

TEST(Tracker, SeedsInStabilizedFrameWithGains) {
  kalman::Config cfg;
  cfg.pan_px_per_rad = 100.0f;
  cfg.tilt_px_per_rad = -100.0f;
  Tracker t(cfg);

  // The seed frame's encoder angles *define* the stabilised frame, so the
  // shift is zero there and the stabilised position equals the detection.
  EXPECT_TRUE(t.step(0.2f, 0.1f, Detection{500.0f, 400.0f, 5.0f}));
  EXPECT_TRUE(t.position().isApprox(Eigen::Vector2f(500.0f, 400.0f), 1e-3f));

  // The image-space ROI at the reference orientation maps back to the
  // detection's image position.
  const kalman::Rect r = t.roi(0.2f, 0.1f);
  EXPECT_NEAR(r.cu, 500.0f, 1e-3f);
  EXPECT_NEAR(r.cv, 400.0f, 1e-3f);
}

TEST(Tracker, TracksStraightLine) {
  Noise n(1);
  Truth truth;
  truth.pos = Eigen::Vector2f(400.0f, 300.0f);
  truth.vel = Eigen::Vector2f(80.0f, -40.0f);

  Tracker t;
  EXPECT_TRUE(t.step(0.0f, 0.0f, noisyDetection(truth, 5.0f, n)));

  float error_sum = 0.0f;
  int k = 0;
  for (; k < 160; ++k) {
    truth.advance(kDt);
    static_cast<void>(t.step(0.0f, 0.0f, noisyDetection(truth, 5.0f, n)));
    if (k >= 120) {
      error_sum += (t.position() - truth.pos).norm();
    }
  }
  EXPECT_LT(error_sum / 40.0f, 6.0f);
  EXPECT_LT((t.position() - truth.pos).norm(), 15.0f);
  // Velocity noise floor: 5 px measurements at 40 Hz leave ~30-60 px/s of
  // residual jitter (harmless — it is sub-pixel over one frame of lead).
  EXPECT_LT((t.velocity() - truth.vel).norm(), 80.0f);
}

TEST(Tracker, TracksConstantAccelerationWithoutLag) {
  Noise n(2);
  Truth truth;
  truth.pos = Eigen::Vector2f(200.0f, 500.0f);
  truth.vel = Eigen::Vector2f(-30.0f, 60.0f);
  truth.acc = Eigen::Vector2f(150.0f, -100.0f);

  Tracker t;
  EXPECT_TRUE(t.step(0.0f, 0.0f, noisyDetection(truth, 5.0f, n)));

  float error_sum = 0.0f;
  for (int k = 0; k < 200; ++k) {
    truth.advance(kDt);
    static_cast<void>(t.step(0.0f, 0.0f, noisyDetection(truth, 5.0f, n)));
    if (k >= 160) {
      error_sum += (t.position() - truth.pos).norm();
    }
  }
  // A CA model tracks a truly constant acceleration with zero steady-state
  // lag; only measurement noise remains.
  EXPECT_LT(error_sum / 40.0f, 8.0f);
  EXPECT_LT((t.velocity() - truth.vel).norm(), 80.0f);
}

TEST(Tracker, GatesFalseDetection) {
  Noise n(3);
  Truth truth;
  truth.pos = Eigen::Vector2f(600.0f, 350.0f);
  truth.vel = Eigen::Vector2f(50.0f, 20.0f);

  Tracker t;
  EXPECT_TRUE(t.step(0.0f, 0.0f, noisyDetection(truth, 5.0f, n)));
  for (int k = 0; k < 60; ++k) {
    truth.advance(kDt);
    EXPECT_TRUE(t.step(0.0f, 0.0f, noisyDetection(truth, 5.0f, n)));
  }

  // 134 px away: accepted at sigma=5 the NIS would be ~134²/(25+σ_pos²) >> 9.21.
  Detection outlier{truth.pos.x() + 120.0f, truth.pos.y() + 60.0f, 5.0f};
  truth.advance(kDt);
  EXPECT_FALSE(t.step(0.0f, 0.0f, outlier));
  EXPECT_GT(t.lastNis(), 9.21f);
  EXPECT_TRUE(t.coasting());
  // The state did not jump to the outlier.
  EXPECT_LT((t.position() - truth.pos).norm(), 40.0f);

  // The track recovers on the next clean detection.
  EXPECT_TRUE(t.step(0.0f, 0.0f, noisyDetection(truth, 5.0f, n)));
  EXPECT_FALSE(t.coasting());
}

TEST(Tracker, WideDetectionSigmaWidensGate) {
  Noise n(4);
  Truth truth;
  truth.pos = Eigen::Vector2f(300.0f, 300.0f);
  truth.vel = Eigen::Vector2f::Zero();

  Tracker tight;
  Tracker loose;
  EXPECT_TRUE(tight.step(0.0f, 0.0f, noisyDetection(truth, 5.0f, n)));
  EXPECT_TRUE(loose.step(0.0f, 0.0f, noisyDetection(truth, 5.0f, n)));
  for (int k = 0; k < 50; ++k) {
    truth.advance(kDt);
    const Detection d = noisyDetection(truth, 5.0f, n);
    static_cast<void>(tight.step(0.0f, 0.0f, d));
    static_cast<void>(loose.step(0.0f, 0.0f, d));
  }

  truth.advance(kDt);
  const float off_u = truth.pos.x() + 40.0f;
  const float off_v = truth.pos.y();
  // Same offset, same frame: sigma=5 rejects, sigma=40 (explicit) accepts.
  EXPECT_FALSE(tight.step(0.0f, 0.0f, Detection{off_u, off_v, 5.0f}));
  EXPECT_TRUE(loose.step(0.0f, 0.0f, Detection{off_u, off_v, 40.0f}));
}

TEST(Tracker, CoastsThroughMissesAndGrowsRoi) {
  Noise n(5);
  Truth truth;
  truth.pos = Eigen::Vector2f(500.0f, 250.0f);
  truth.vel = Eigen::Vector2f(70.0f, -35.0f);

  Tracker t;
  EXPECT_TRUE(t.step(0.0f, 0.0f, noisyDetection(truth, 5.0f, n)));
  for (int k = 0; k < 80; ++k) {
    truth.advance(kDt);
    static_cast<void>(t.step(0.0f, 0.0f, noisyDetection(truth, 5.0f, n)));
  }
  const float var_before = t.covariance()(0, 0);
  const float half_before = t.roi(0.0f, 0.0f).half_w;

  for (int k = 0; k < 20; ++k) {
    truth.advance(kDt);
    EXPECT_FALSE(t.step(0.0f, 0.0f));  // detector miss: coast
    EXPECT_TRUE(t.coasting());
    EXPECT_GT(t.framesSinceUpdate(), 0);
  }
  // Uncertainty and search window both grew; position is still on target.
  EXPECT_GT(t.covariance()(0, 0), var_before);
  EXPECT_GT(t.roi(0.0f, 0.0f).half_w, half_before);
  EXPECT_LT((t.position() - truth.pos).norm(), 45.0f);
  EXPECT_GE(t.roi(0.0f, 0.0f).half_w, t.config().roi_min_half_extent);

  // Recovery: the next detection is accepted and the coast flag clears.
  EXPECT_TRUE(t.step(0.0f, 0.0f, noisyDetection(truth, 5.0f, n)));
  EXPECT_FALSE(t.coasting());
  EXPECT_EQ(t.framesSinceUpdate(), 0);
}

TEST(Tracker, CompensatesPanTiltSlew) {
  kalman::Config cfg;
  cfg.pan_px_per_rad = 120.0f;
  cfg.tilt_px_per_rad = -120.0f;
  Tracker t(cfg);

  // The drone is stationary *in the world*; the gimbal slews, so its image
  // position moves while its stabilised position must not.
  const Eigen::Vector2f world_pos(500.0f, 400.0f);
  Noise n(6);
  const float pan_rate = 0.008f;  // rad/frame ≈ 0.32 rad/s
  const float tilt_rate = -0.004f;

  for (int k = 0; k < 100; ++k) {
    const float pan = pan_rate * static_cast<float>(k);
    const float tilt = tilt_rate * static_cast<float>(k);
    const Eigen::Vector2f shift =
        Eigen::Vector2f(cfg.pan_px_per_rad * pan, cfg.tilt_px_per_rad * tilt);
    const Detection d{world_pos.x() + shift.x() + 5.0f * n(),
                      world_pos.y() + shift.y() + 5.0f * n(), 5.0f};
    static_cast<void>(t.step(pan, tilt, d));

    if (k >= 60) {
      // Stabilised estimate stays on the stationary target...
      EXPECT_LT((t.position() - world_pos).norm(), 15.0f) << "frame " << k;
      // ...so the filter does not mistake gimbal motion for target motion.
      EXPECT_LT(t.velocity().norm(), 80.0f) << "frame " << k;
      // ...and the image-space ROI follows the gimbal.
      const kalman::Rect r = t.roi(pan, tilt);
      EXPECT_NEAR(r.cu, world_pos.x() + cfg.pan_px_per_rad * pan, 15.0f);
      EXPECT_NEAR(r.cv, world_pos.y() + cfg.tilt_px_per_rad * tilt, 15.0f);
    }
  }
}

TEST(Tracker, PredictExtrapolatesLatency) {
  Noise n(7);
  Truth truth;
  truth.pos = Eigen::Vector2f(250.0f, 450.0f);
  truth.vel = Eigen::Vector2f(60.0f, 30.0f);

  Tracker t;
  EXPECT_TRUE(t.step(0.0f, 0.0f, noisyDetection(truth, 5.0f, n)));
  for (int k = 0; k < 80; ++k) {
    truth.advance(kDt);
    static_cast<void>(t.step(0.0f, 0.0f, noisyDetection(truth, 5.0f, n)));
  }

  // predict() is const: lead the state by a known detector latency without
  // disturbing the filter.
  const float horizon = 0.1f;
  const kalman::PredictedState p = t.predict(horizon);
  const Eigen::Vector2f future = truth.pos + truth.vel * horizon;
  EXPECT_LT((p.position - future).norm(), 20.0f);
  EXPECT_LT((p.velocity - truth.vel).norm(), 60.0f);
  // The original state is untouched.
  EXPECT_LT((t.position() - truth.pos).norm(), 15.0f);
}

TEST(Tracker, RoiRespectsFloorAndGrowsWithHorizon) {
  Noise n(8);
  Truth truth;
  truth.pos = Eigen::Vector2f(100.0f, 100.0f);

  Tracker t;
  EXPECT_TRUE(t.step(0.0f, 0.0f, noisyDetection(truth, 5.0f, n)));
  for (int k = 0; k < 40; ++k) {
    truth.advance(kDt);
    static_cast<void>(t.step(0.0f, 0.0f, noisyDetection(truth, 5.0f, n)));
  }

  const kalman::Rect now = t.roi(0.0f, 0.0f);
  const kalman::Rect far = t.roi(0.0f, 0.0f, 2.0f);
  EXPECT_GE(now.half_w, t.config().roi_min_half_extent);
  EXPECT_GE(now.half_h, t.config().roi_min_half_extent);
  EXPECT_GT(far.half_w, now.half_w);
  EXPECT_GT(far.half_h, now.half_h);
}

TEST(Tracker, NeesConsistentUnderMatchingModel) {
  // Monte-Carlo NEES against a truth process generated from the *same* model
  // and noise parameters.  Gating is disabled so the filter is an exact KF
  // (gating conditions the update and biases the statistic).
  constexpr int kRuns = 100;
  constexpr int kSteps = 100;
  constexpr float kMeasSigma = 5.0f;

  kalman::Config cfg;
  cfg.gate_chi2 = 1.0e9f;

  Noise n(9);
  double nees_sum = 0.0;
  int samples = 0;

  for (int run = 0; run < kRuns; ++run) {
    Tracker t(cfg);

    // Prior sample x0 ~ N(0, P0); the seeded estimate carries error exactly
    // N(0, P0) (position error from the first detection's noise, velocity and
    // acceleration error from the uninitialised zeros).
    Eigen::Matrix<float, 6, 1> x_true;
    for (int i = 0; i < 6; ++i) {
      const float sigma = (i < 2)   ? cfg.init_pos_sigma
                          : (i < 4) ? cfg.init_vel_sigma
                                    : cfg.init_acc_sigma;
      x_true(i) = sigma * n();
    }
    Detection d0{x_true(0) + cfg.init_pos_sigma * n(),
                 x_true(1) + cfg.init_pos_sigma * n(), kMeasSigma};
    static_cast<void>(t.step(0.0f, 0.0f, d0));

    const Eigen::Matrix<float, 6, 6> F = kalman::detail::caTransition(kDt);
    const Eigen::Matrix<float, 6, 6> Q =
        kalman::detail::caNoise(kDt, cfg.q_jerk);
    const Eigen::LLT<Eigen::Matrix<float, 6, 6>> llt(Q);

    for (int k = 0; k < kSteps; ++k) {
      Eigen::Matrix<float, 6, 1> w;
      for (int i = 0; i < 6; ++i) {
        w(i) = n();
      }
      x_true = F * x_true + llt.matrixL() * w;

      const Detection d{x_true(0) + kMeasSigma * n(),
                        x_true(1) + kMeasSigma * n(), kMeasSigma};
      static_cast<void>(t.step(0.0f, 0.0f, d));

      const Eigen::Matrix<float, 6, 1> e = x_true - t.state();
      nees_sum += e.dot(t.covariance().ldlt().solve(e));
      ++samples;
    }
  }

  // E[NEES] = 6 (state dim); std of the mean ≈ 6·sqrt(2/N) ≈ 0.085.
  const double mean_nees = nees_sum / static_cast<double>(samples);
  EXPECT_NEAR(mean_nees, 6.0, 0.35) << "samples=" << samples;
}

// ---------------------------------------------------------------------------
// Runtime input validation
// ---------------------------------------------------------------------------

TEST(Tracker, LargerDtOverrideGrowsCovarianceFaster) {
  Tracker nominal;
  Tracker stretched;
  EXPECT_TRUE(nominal.step(0.0f, 0.0f, Detection{400.0f, 300.0f, 5.0f}));
  EXPECT_TRUE(stretched.step(0.0f, 0.0f, Detection{400.0f, 300.0f, 5.0f}));

  // Seeded at rest, so both stay put in position; the state covariance must
  // spread faster over the longer prediction interval.
  for (int k = 0; k < 20; ++k) {
    static_cast<void>(nominal.step(0.0f, 0.0f));
    static_cast<void>(stretched.step(0.0f, 0.0f, std::nullopt, 2.0f * kDt));
  }

  EXPECT_FLOAT_EQ(nominal.position().x(), stretched.position().x());
  EXPECT_FLOAT_EQ(nominal.position().y(), stretched.position().y());
  EXPECT_GT(stretched.covariance()(0, 0), nominal.covariance()(0, 0));
  EXPECT_GT(stretched.covariance()(2, 2), nominal.covariance()(2, 2));
}

TEST(Tracker, InvalidDtFallsBackToConfigDt) {
  Tracker reference;
  EXPECT_TRUE(reference.step(0.0f, 0.0f, Detection{400.0f, 300.0f, 5.0f}));
  for (int k = 0; k < 10; ++k) {
    static_cast<void>(reference.step(0.0f, 0.0f));
  }

  for (const float bad :
       {0.0f, -0.1f, std::nanf(""), std::numeric_limits<float>::infinity()}) {
    Tracker t;
    EXPECT_TRUE(t.step(0.0f, 0.0f, Detection{400.0f, 300.0f, 5.0f}));
    for (int k = 0; k < 10; ++k) {
      static_cast<void>(t.step(0.0f, 0.0f, std::nullopt, bad));
    }
    EXPECT_FLOAT_EQ(t.covariance()(0, 0), reference.covariance()(0, 0))
        << "dt=" << bad;
    EXPECT_FLOAT_EQ(t.position().x(), reference.position().x());
  }
}

TEST(Tracker, NonFiniteEncoderDropsMeasurementButKeepsClock) {
  Tracker t;
  EXPECT_TRUE(t.step(0.0f, 0.0f, Detection{400.0f, 300.0f, 5.0f}));
  const Eigen::Vector2f before = t.position();

  const float nan = std::nanf("");
  // A detection cannot be mapped into the stabilised frame without valid
  // encoder angles: dropped, but the frame still advances.
  EXPECT_FALSE(t.step(nan, 0.0f, Detection{400.0f, 300.0f, 5.0f}));
  EXPECT_TRUE(t.coasting());
  EXPECT_EQ(t.framesSinceUpdate(), 1);
  EXPECT_TRUE(t.position().isApprox(before, 0.0f));

  // The next clean frame works again (the unwrap history was left alone).
  EXPECT_TRUE(t.step(0.0f, 0.0f, Detection{400.0f, 300.0f, 5.0f}));
  EXPECT_FALSE(t.coasting());
}

TEST(Tracker, NonFiniteDetectionIsGatedOut) {
  Tracker t;
  EXPECT_TRUE(t.step(0.0f, 0.0f, Detection{400.0f, 300.0f, 5.0f}));

  const float nan = std::nanf("");
  EXPECT_FALSE(t.step(0.0f, 0.0f, Detection{nan, nan, 5.0f}));
  EXPECT_TRUE(t.coasting());
  // Seeded at rest with no accepted measurement since: prediction leaves the
  // position bit-identical, so the NaN never reached the state.
  EXPECT_FLOAT_EQ(t.position().x(), 400.0f);
  EXPECT_FLOAT_EQ(t.position().y(), 300.0f);
  EXPECT_TRUE(std::isinf(t.lastNis()));
}

TEST(Tracker, NonFiniteSeedIsRejectedAndLeavesTrackIntact) {
  Tracker t;
  const float nan = std::nanf("");

  EXPECT_FALSE(t.reset(Detection{nan, 300.0f, 5.0f}, 0.0f, 0.0f));
  EXPECT_FALSE(t.initialized());
  EXPECT_FALSE(t.step(nan, 0.0f, Detection{400.0f, 300.0f, 5.0f}));
  EXPECT_FALSE(t.initialized());
  EXPECT_FALSE(t.reset(Detection{400.0f, 300.0f, 5.0f}, nan, 0.0f));
  EXPECT_FALSE(t.initialized());

  EXPECT_TRUE(t.reset(Detection{400.0f, 300.0f, 5.0f}, 0.0f, 0.0f));
  // A garbage re-seed must not damage the live track.
  EXPECT_FALSE(t.reset(Detection{400.0f, nan, 5.0f}, 0.0f, 0.0f));
  EXPECT_TRUE(t.position().isApprox(Eigen::Vector2f(400.0f, 300.0f), 1e-3f));
  EXPECT_TRUE(t.initialized());
}

TEST(Tracker, InvalidConfigFieldsFallBackToDefaults) {
  Config bad;
  bad.dt = 0.0f;
  bad.q_jerk = std::nanf("");
  bad.detection_sigma = -1.0f;
  bad.gate_chi2 = std::numeric_limits<float>::infinity();
  bad.init_pos_sigma = -5.0f;
  bad.init_vel_sigma = std::nanf("");
  bad.roi_k_sigma = 0.0f;
  bad.roi_min_half_extent = std::nanf("");
  bad.pan_px_per_rad = std::nanf("");
  bad.tilt_px_per_rad = std::numeric_limits<float>::infinity();
  bad.max_coast_frames = 7;  // integer: no NaN possible, passes through

  Tracker t(bad);
  const Config& c = t.config();
  EXPECT_FLOAT_EQ(c.dt, Config().dt);
  EXPECT_FLOAT_EQ(c.q_jerk, Config().q_jerk);
  EXPECT_FLOAT_EQ(c.detection_sigma, Config().detection_sigma);
  EXPECT_FLOAT_EQ(c.gate_chi2, Config().gate_chi2);
  EXPECT_FLOAT_EQ(c.init_pos_sigma, Config().init_pos_sigma);
  EXPECT_FLOAT_EQ(c.init_vel_sigma, Config().init_vel_sigma);
  EXPECT_FLOAT_EQ(c.roi_k_sigma, Config().roi_k_sigma);
  EXPECT_FLOAT_EQ(c.roi_min_half_extent, Config().roi_min_half_extent);
  EXPECT_FLOAT_EQ(c.pan_px_per_rad, 0.0f);
  EXPECT_FLOAT_EQ(c.tilt_px_per_rad, 0.0f);
  EXPECT_EQ(c.max_coast_frames, 7);

  // Valid fields survive untouched, and the tracker runs on the sanitised set.
  Config ok;
  ok.dt = 0.01f;
  ok.q_jerk = 1234.0f;
  ok.pan_px_per_rad = 55.0f;
  const Tracker t2(ok);
  EXPECT_FLOAT_EQ(t2.config().dt, 0.01f);
  EXPECT_FLOAT_EQ(t2.config().q_jerk, 1234.0f);
  EXPECT_FLOAT_EQ(t2.config().pan_px_per_rad, 55.0f);
  Tracker t3(bad);
  EXPECT_TRUE(t3.step(0.0f, 0.0f, Detection{400.0f, 300.0f, 5.0f}));
}

// ---------------------------------------------------------------------------
// ROI knobs
// ---------------------------------------------------------------------------

TEST(Tracker, RoiKSigmaScalesHalfExtentAndInvalidFallsBack) {
  Tracker t;
  EXPECT_TRUE(t.step(0.0f, 0.0f, Detection{400.0f, 300.0f, 5.0f}));
  static_cast<void>(t.step(0.0f, 0.0f, Detection{400.0f, 300.0f, 5.0f}));

  // With σ ≈ 5 px both multipliers clear the 40 px floor, so the half-extent
  // must scale exactly with k.
  const kalman::Rect wide = t.roi(0.0f, 0.0f, 0.0f, 40.0f);
  const kalman::Rect base = t.roi(0.0f, 0.0f, 0.0f, 20.0f);
  EXPECT_GT(base.half_w, t.config().roi_min_half_extent);
  EXPECT_NEAR(wide.half_w, 2.0f * base.half_w, 1e-3f);
  EXPECT_NEAR(wide.half_h, 2.0f * base.half_h, 1e-3f);

  const kalman::Rect dflt = t.roi(0.0f, 0.0f);
  for (const float bad : {0.0f, -3.0f, std::nanf("")}) {
    const kalman::Rect r = t.roi(0.0f, 0.0f, 0.0f, bad);
    EXPECT_FLOAT_EQ(r.half_w, dflt.half_w) << "k=" << bad;
    EXPECT_FLOAT_EQ(r.half_h, dflt.half_h) << "k=" << bad;
  }
}

TEST(Tracker, NonPositiveDetectionSigmaFallsBackToDefault) {
  // Three trackers fed bit-identical detections except for the reported
  // sigma: 0 (invalid → Config::detection_sigma = 5), NaN (invalid → 5), and
  // an explicit 5.0f.  All three must take every accept/reject decision
  // identically and end up in the same state.
  Tracker defaulted;
  Tracker nan_sigma;
  Tracker explicit_sigma;

  Noise n(10);
  Truth truth;
  truth.pos = Eigen::Vector2f(350.0f, 350.0f);

  const Detection d0{truth.pos.x(), truth.pos.y(), 0.0f};
  const Detection n0{truth.pos.x(), truth.pos.y(), std::nanf("")};
  const Detection e0{truth.pos.x(), truth.pos.y(), 5.0f};
  EXPECT_TRUE(defaulted.step(0.0f, 0.0f, d0));
  EXPECT_TRUE(nan_sigma.step(0.0f, 0.0f, n0));
  EXPECT_TRUE(explicit_sigma.step(0.0f, 0.0f, e0));

  for (int k = 0; k < 40; ++k) {
    truth.advance(kDt);
    const Detection d{truth.pos.x() + n(), truth.pos.y() + n(), 0.0f};
    const Detection v{d.u, d.v, std::nanf("")};
    const Detection e{d.u, d.v, 5.0f};
    const bool a = defaulted.step(0.0f, 0.0f, d);
    const bool b = nan_sigma.step(0.0f, 0.0f, v);
    const bool c = explicit_sigma.step(0.0f, 0.0f, e);
    ASSERT_EQ(a, b);
    ASSERT_EQ(a, c);
  }

  // One near-boundary outlier: same decision, same NIS, same state.
  truth.advance(kDt);
  const Detection o{truth.pos.x() + 40.0f, truth.pos.y(), 0.0f};
  const bool a = defaulted.step(0.0f, 0.0f, o);
  EXPECT_FALSE(nan_sigma.step(0.0f, 0.0f, Detection{o.u, o.v, std::nanf("")}));
  EXPECT_FALSE(explicit_sigma.step(0.0f, 0.0f, Detection{o.u, o.v, 5.0f}));
  EXPECT_FALSE(a);

  EXPECT_FLOAT_EQ(defaulted.position().x(), explicit_sigma.position().x());
  EXPECT_FLOAT_EQ(defaulted.position().y(), explicit_sigma.position().y());
  EXPECT_FLOAT_EQ(defaulted.position().x(), nan_sigma.position().x());
  EXPECT_FLOAT_EQ(defaulted.lastNis(), explicit_sigma.lastNis());
  EXPECT_FLOAT_EQ(defaulted.covariance()(0, 0),
                  explicit_sigma.covariance()(0, 0));
}

// ---------------------------------------------------------------------------
// Re-seeding and track loss
// ---------------------------------------------------------------------------

TEST(Tracker, ResetReSeedsLiveTrack) {
  Tracker t;
  EXPECT_TRUE(t.step(0.0f, 0.0f, Detection{400.0f, 300.0f, 5.0f}));
  // An accepted measurement so lastNis is live (2 px off an uninitialised
  // velocity: NIS is small, so it is accepted).
  EXPECT_TRUE(t.step(0.0f, 0.0f, Detection{410.0f, 300.0f, 5.0f}));
  EXPECT_GT(t.lastNis(), 0.0f);
  static_cast<void>(t.step(0.0f, 0.0f));
  static_cast<void>(t.step(0.0f, 0.0f));
  EXPECT_TRUE(t.coasting());

  EXPECT_TRUE(t.reset(Detection{900.0f, 700.0f, 5.0f}, 0.5f, -0.2f));
  EXPECT_TRUE(t.position().isApprox(Eigen::Vector2f(900.0f, 700.0f), 1e-3f));
  EXPECT_TRUE(t.velocity().isZero());
  EXPECT_EQ(t.framesSinceUpdate(), 0);
  EXPECT_FALSE(t.coasting());
  EXPECT_FALSE(t.lost());
  // Priors restored, and the stale NIS went with the old track.
  EXPECT_FLOAT_EQ(t.covariance()(0, 0), 30.0f * 30.0f);
  EXPECT_FLOAT_EQ(t.covariance()(2, 2), 500.0f * 500.0f);
  EXPECT_FLOAT_EQ(t.lastNis(), 0.0f);
}

TEST(Tracker, TrackIsDeclaredLostAfterMaxCoastAndReSeeds) {
  Config cfg;
  cfg.max_coast_frames = 10;
  Tracker t(cfg);
  EXPECT_TRUE(t.step(0.0f, 0.0f, Detection{400.0f, 300.0f, 5.0f}));
  static_cast<void>(t.step(0.0f, 0.0f, Detection{402.0f, 300.0f, 5.0f}));

  for (int k = 0; k < 9; ++k) {
    EXPECT_FALSE(t.step(0.0f, 0.0f));
    EXPECT_FALSE(t.lost()) << "frame " << k;
  }
  EXPECT_EQ(t.framesSinceUpdate(), 9);

  // The 10th consecutive miss crosses the threshold.
  EXPECT_FALSE(t.step(0.0f, 0.0f));
  EXPECT_TRUE(t.lost());
  EXPECT_TRUE(t.coasting());
  EXPECT_EQ(t.framesSinceUpdate(), 10);

  // While lost the track is frozen: no further coast growth, no prediction.
  const Eigen::Vector2f frozen = t.position();
  EXPECT_FALSE(t.step(0.0f, 0.0f));
  EXPECT_EQ(t.framesSinceUpdate(), 10);
  EXPECT_TRUE(t.position().isApprox(frozen, 0.0f));

  // A detection re-seeds instead of blending into the stale state.
  EXPECT_TRUE(t.step(0.0f, 0.0f, Detection{900.0f, 700.0f, 5.0f}));
  EXPECT_FALSE(t.lost());
  EXPECT_FALSE(t.coasting());
  EXPECT_TRUE(t.position().isApprox(Eigen::Vector2f(900.0f, 700.0f), 1e-3f));
}

TEST(Tracker, CoastLossDetectionCanBeDisabled) {
  Config cfg;
  cfg.max_coast_frames = 0;  // <= 0 disables: coast as long as it takes
  Tracker t(cfg);
  EXPECT_TRUE(t.step(0.0f, 0.0f, Detection{400.0f, 300.0f, 5.0f}));

  for (int k = 0; k < 300; ++k) {
    EXPECT_FALSE(t.step(0.0f, 0.0f));
    EXPECT_FALSE(t.lost()) << "frame " << k;
  }
  EXPECT_EQ(t.framesSinceUpdate(), 300);
  EXPECT_TRUE(t.coasting());
}

// ---------------------------------------------------------------------------
// Consistency statistics
// ---------------------------------------------------------------------------

TEST(Tracker, NisConsistentUnderMatchingModel) {
  // Monte-Carlo NIS against a truth process generated from the *same* model
  // and noise parameters, with gating disabled so the update is an exact KF.
  // E[NIS] = measurement dimension = 2.
  constexpr int kRuns = 100;
  constexpr int kSteps = 100;
  constexpr float kMeasSigma = 5.0f;

  kalman::Config cfg;
  cfg.gate_chi2 = 1.0e9f;

  Noise n(11);
  double nis_sum = 0.0;
  int samples = 0;

  for (int run = 0; run < kRuns; ++run) {
    Tracker t(cfg);

    // Prior sample x0 ~ N(0, P0); the seeded estimate carries error exactly
    // N(0, P0).
    Eigen::Matrix<float, 6, 1> x_true;
    for (int i = 0; i < 6; ++i) {
      const float sigma = (i < 2)   ? cfg.init_pos_sigma
                          : (i < 4) ? cfg.init_vel_sigma
                                    : cfg.init_acc_sigma;
      x_true(i) = sigma * n();
    }
    const Detection d0{x_true(0) + cfg.init_pos_sigma * n(),
                       x_true(1) + cfg.init_pos_sigma * n(), kMeasSigma};
    EXPECT_TRUE(t.step(0.0f, 0.0f, d0));

    const Eigen::Matrix<float, 6, 6> F = kalman::detail::caTransition(kDt);
    const Eigen::Matrix<float, 6, 6> Q =
        kalman::detail::caNoise(kDt, cfg.q_jerk);
    const Eigen::LLT<Eigen::Matrix<float, 6, 6>> llt(Q);

    for (int k = 0; k < kSteps; ++k) {
      Eigen::Matrix<float, 6, 1> w;
      for (int i = 0; i < 6; ++i) {
        w(i) = n();
      }
      x_true = F * x_true + llt.matrixL() * w;

      const Detection d{x_true(0) + kMeasSigma * n(),
                        x_true(1) + kMeasSigma * n(), kMeasSigma};
      EXPECT_TRUE(t.step(0.0f, 0.0f, d));
      EXPECT_GT(t.lastNis(), 0.0f);

      nis_sum += t.lastNis();
      ++samples;
    }
  }

  // E[NIS] = 2; std of the mean for independent chi²₂ samples ≈ 0.028, and
  // the tolerance is widened for the serial correlation of consecutive
  // innovations (same reasoning as the NEES test above).
  const double mean_nis = nis_sum / static_cast<double>(samples);
  EXPECT_NEAR(mean_nis, 2.0, 0.3) << "samples=" << samples;
}

TEST(Tracker, MatchesDenseJosephReferenceWithVariableTimingAndNoise) {
  using Matrix6d = Eigen::Matrix<double, 6, 6>;
  using Vector6d = Eigen::Matrix<double, 6, 1>;
  Eigen::Matrix<double, 2, 6> H = Eigen::Matrix<double, 2, 6>::Zero();
  H(0, 0) = H(1, 1) = 1.0;
  for (const float q : {0.0f, 2.0e5f}) {
    Config cfg;
    cfg.q_jerk = q;
    cfg.init_pos_sigma = 7.0f;
    cfg.init_vel_sigma = 80.0f;
    cfg.init_acc_sigma = 300.0f;
    cfg.roi_min_half_extent = 0.0f;
    cfg.max_coast_frames = 0;
    Tracker t(cfg);
    ASSERT_TRUE(t.reset({170.0f, -80.0f, 3.0f}, 0.0f, 0.0f));
    Vector6d x = Vector6d::Zero();
    x.head<2>() << 170.0, -80.0;
    Matrix6d p = Matrix6d::Zero();
    p.diagonal() << 49.0, 49.0, 6400.0, 6400.0, 90000.0, 90000.0;
    Noise noise(93);
    double time = 0.0;
    int accepted_count = 0;
    int rejected_count = 0;
    for (int frame = 0; frame < 500; ++frame) {
      SCOPED_TRACE(::testing::Message() << "q=" << q << " frame=" << frame);
      const float dt = frame % 13 == 0 ? 0.12f : 0.017f;
      time += dt;
      const Matrix6d F = kalman::detail::caTransition(dt).cast<double>();
      const Matrix6d Q = kalman::detail::caNoise(dt, q).cast<double>();
      x = F * x;
      p = F * p * F.transpose() + Q;
      std::optional<Detection> det;
      bool accepted = false;
      if (frame % 7 != 0) {
        const float sigma = frame % 3 == 0 ? 1.5f : 8.0f;
        det = Detection{
            static_cast<float>(170.0 + 23.0 * time) + sigma * noise(),
            static_cast<float>(-80.0 - 11.0 * time) + sigma * noise(), sigma};
        if (frame % 19 == 0) {
          det->v += 1000.0f;
        }
        const double r = static_cast<double>(sigma) * sigma;
        const Eigen::Vector2d z(det->u, det->v);
        const Eigen::Vector2d nu = z - H * x;
        const Eigen::Matrix2d s =
            H * p * H.transpose() + r * Eigen::Matrix2d::Identity();
        const double nis = nu.dot(s.ldlt().solve(nu));
        accepted = nis < cfg.gate_chi2;
        if (accepted) {
          const Eigen::Matrix<double, 6, 2> gain =
              p * H.transpose() * s.ldlt().solve(Eigen::Matrix2d::Identity());
          x += gain * nu;
          const Matrix6d a = Matrix6d::Identity() - gain * H;
          p = a * p * a.transpose() + r * gain * gain.transpose();
          ++accepted_count;
        } else {
          ++rejected_count;
        }
        EXPECT_EQ(t.step(0.0f, 0.0f, det, dt), accepted);
        EXPECT_NEAR(t.lastNis(), nis, 0.002 + 0.001 * nis);
      } else {
        EXPECT_FALSE(t.step(0.0f, 0.0f, std::nullopt, dt));
      }
      EXPECT_TRUE(t.state().cast<double>().isApprox(x, 2e-5));
      const Matrix6d actual = t.covariance().cast<double>();
      for (int row = 0; row < 6; ++row) {
        for (int col = 0; col < 6; ++col) {
          EXPECT_NEAR(actual(row, col), p(row, col),
                      0.0002 + 0.0003 * std::abs(p(row, col)));
        }
      }
      for (const float horizon : {0.0f, 0.031f, 0.4f}) {
        const Matrix6d lead_f =
            kalman::detail::caTransition(horizon).cast<double>();
        const Matrix6d lead_p =
            lead_f * p * lead_f.transpose() +
            kalman::detail::caNoise(horizon, q).cast<double>();
        const Vector6d lead_x = lead_f * x;
        const auto lead = t.predict(horizon);
        EXPECT_TRUE(
            lead.position.cast<double>().isApprox(lead_x.head<2>(), 2e-5));
        // Extrapolation can cancel velocity against acceleration near zero;
        // use an absolute floor as well as a relative float tolerance.
        EXPECT_LE((lead.velocity.cast<double>() - lead_x.segment<2>(2)).norm(),
                  0.001 + 2e-4 * lead_x.segment<2>(2).norm());
        EXPECT_TRUE(lead.covariance.cast<double>().isApprox(lead_p, 3e-4));
        const auto roi = t.roi(0.0f, 0.0f, horizon);
        EXPECT_NEAR(roi.cu, lead_x(0), 0.003);
        EXPECT_NEAR(roi.cv, lead_x(1), 0.003);
        EXPECT_NEAR(roi.half_w, 3.0 * std::sqrt(lead_p(0, 0)), 0.003);
        EXPECT_NEAR(roi.half_h, 3.0 * std::sqrt(lead_p(1, 1)), 0.003);
      }
    }
    EXPECT_GT(accepted_count, 300);
    EXPECT_GT(rejected_count, 10);
  }
}

TEST(Tracker, GateIsJointAndRejectsEquality) {
  Config cfg;
  cfg.q_jerk = 0.0f;
  cfg.init_pos_sigma = cfg.init_vel_sigma = cfg.init_acc_sigma = 0.0f;
  cfg.gate_chi2 = 5.0f;
  Tracker t(cfg);
  ASSERT_TRUE(t.reset({0.0f, 0.0f, 2.0f}, 0.0f, 0.0f));
  // Each axis alone passes, but the joint NIS is exactly the threshold.
  EXPECT_FALSE(t.step(0.0f, 0.0f, Detection{4.0f, 2.0f, 2.0f}));
  EXPECT_FLOAT_EQ(t.lastNis(), 5.0f);
  EXPECT_TRUE(t.state().isZero());
  EXPECT_TRUE(t.covariance().isZero());
  EXPECT_TRUE(t.step(0.0f, 0.0f, Detection{4.0f, 1.0f, 2.0f}));
  EXPECT_FLOAT_EQ(t.lastNis(), 4.25f);
}
