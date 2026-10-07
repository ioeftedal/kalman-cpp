#include "kalman/tracker.hpp"

#include <gtest/gtest.h>

#include <cmath>
#include <random>
#include <vector>

#include "kalman/detail/ca_model.hpp"

namespace {

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
    t.step(0.0f, 0.0f, noisyDetection(truth, 5.0f, n));
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
    t.step(0.0f, 0.0f, noisyDetection(truth, 5.0f, n));
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
    tight.step(0.0f, 0.0f, d);
    loose.step(0.0f, 0.0f, d);
  }

  truth.advance(kDt);
  const Detection offset{truth.pos.x() + 40.0f, truth.pos.y(), 0.0f};
  // Same offset, same frame: sigma=5 rejects, sigma=40 (explicit) accepts.
  EXPECT_FALSE(tight.step(0.0f, 0.0f, Detection{offset.u, offset.v, 5.0f}));
  EXPECT_TRUE(loose.step(0.0f, 0.0f, Detection{offset.u, offset.v, 40.0f}));
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
    t.step(0.0f, 0.0f, noisyDetection(truth, 5.0f, n));
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
    t.step(pan, tilt, d);

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
    t.step(0.0f, 0.0f, noisyDetection(truth, 5.0f, n));
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
    t.step(0.0f, 0.0f, noisyDetection(truth, 5.0f, n));
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
    t.step(0.0f, 0.0f, d0);

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
      t.step(0.0f, 0.0f, d);

      const Eigen::Matrix<float, 6, 1> e = x_true - t.state();
      nees_sum += e.dot(t.covariance().ldlt().solve(e));
      ++samples;
    }
  }

  // E[NEES] = 6 (state dim); std of the mean ≈ 6·sqrt(2/N) ≈ 0.085.
  const double mean_nees = nees_sum / static_cast<double>(samples);
  EXPECT_NEAR(mean_nees, 6.0, 0.35) << "samples=" << samples;
}
