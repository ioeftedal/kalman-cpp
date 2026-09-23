/// @file
/// @brief IMM filter tests (CV / CA / CT set): CT kinematics structure,
///        straight-line tracking, maneuver handling vs a single CV filter,
///        outlier gating, and adaptive measurement noise.

#include "kalman/core/imm.hpp"

#include <gtest/gtest.h>

#include <Eigen/Cholesky>
#include <Eigen/Core>
#include <cmath>
#include <random>
#include <vector>

#include "kalman/kinematics/ct_pixel.hpp"
#include "kalman/models/pixel.hpp"

// -----------------------------------------------------------------------
//  Type aliases
// -----------------------------------------------------------------------

using PixelCvFilter = kalman::UnscentedKalmanFilter<double, 4, 2, kalman::PixelPolicy>;
using ImmCvModel = kalman::ImmPixelModel<double, void>;
using ImmCaModel = kalman::ImmPixelModel<double, kalman::PixelCATag>;
using ImmCtModel = kalman::ImmPixelModel<double, kalman::PixelCTTag>;
using Imm = kalman::ImmFilter<double, ImmCvModel, ImmCaModel, ImmCtModel>;

// -----------------------------------------------------------------------
//  Local trajectory generators (4-D common state: u, v, vu, vv)
// -----------------------------------------------------------------------

/// Constant-velocity straight-line states.
std::vector<Imm::VectorC> cvStates(int steps, double dt, double vx, double vy) {
  std::vector<Imm::VectorC> states;
  states.reserve(steps);
  for (int k = 0; k < steps; ++k) {
    Imm::VectorC x;
    x << vx * k * dt, vy * k * dt, vx, vy;
    states.push_back(x);
  }
  return states;
}

/// Constant-turn-rate states (circle radius R = v / w).
std::vector<Imm::VectorC> ctStates(int steps, double dt, double v, double w) {
  const double R = v / w;
  std::vector<Imm::VectorC> states;
  states.reserve(steps);
  for (int k = 0; k < steps; ++k) {
    const double t = k * dt;
    Imm::VectorC x;
    x << R * std::sin(w * t), R * (1.0 - std::cos(w * t)), v * std::cos(w * t), v * std::sin(w * t);
    states.push_back(x);
  }
  return states;
}

/// Drive the IMM over `states` with position measurements whose noise follows
/// `meas_sigma` (defaults to the standard deviation implied by `R`).
/// Returns the per-step normalised innovations (NEES) of the fused state.
std::vector<double> driveIMM(Imm& imm, const std::vector<Imm::VectorC>& states, double dt,
                             const Imm::MatrixM& R, double gate = 9.21, unsigned seed = 7,
                             double meas_sigma = -1.0) {
  const int steps = static_cast<int>(states.size());
  Imm::MatrixC P0 = Imm::MatrixC::Identity() * (80.0 * 80.0);
  P0(2, 2) = 60.0 * 60.0;
  P0(3, 3) = 60.0 * 60.0;
  imm.initialize(states[0], P0);

  const double sigma = meas_sigma >= 0.0 ? meas_sigma : std::sqrt(R(0, 0));
  std::mt19937 rng(seed);
  std::normal_distribution<double> noise(0.0, sigma);

  std::vector<double> nees;
  nees.reserve(steps);
  for (int k = 0; k < steps; ++k) {
    if (k > 0) {
      imm.predict(dt);
    }
    Imm::VectorM z = states[k].template head<2>();
    z(0) += noise(rng);
    z(1) += noise(rng);
    imm.update(z, R, gate);

    const Imm::VectorC err = states[k] - imm.getState();
    nees.push_back(imm.getCovariance().ldlt().solve(err).dot(err));
  }
  return nees;
}

// -----------------------------------------------------------------------
//  CT kinematics structure
// -----------------------------------------------------------------------

TEST(CtKinematics, Structure) {
  using Vec5 = Eigen::Matrix<double, 5, 1>;
  Vec5 x;
  x << 100.0, 200.0, 50.0, 30.0, 0.6;

  auto F0 = kalman::kinematics::computeF_pixel_CT(x, 0.0);
  ASSERT_EQ(F0.rows(), 5);
  ASSERT_EQ(F0.cols(), 5);
  EXPECT_TRUE(F0.isApprox(Eigen::Matrix<double, 5, 5>::Identity(), 1e-9))
      << "F(x, 0) should be identity";

  auto Fdt = kalman::kinematics::computeF_pixel_CT(x, 0.1);
  EXPECT_FALSE(Fdt.isIdentity()) << "F(x, dt>0) should not be identity";
  EXPECT_TRUE(Fdt.allFinite());

  auto Q = kalman::kinematics::computeQ_pixel_CT(0.1, 3000.0);
  EXPECT_TRUE(Q.isApprox(Q.transpose(), 1e-10)) << "Q should be symmetric";
  EXPECT_TRUE(Q.allFinite());

  Eigen::SelfAdjointEigenSolver<Eigen::Matrix<double, 5, 5>> eig(Q);
  for (int i = 0; i < 5; ++i) {
    EXPECT_GE(eig.eigenvalues()(i), -1e-6)
        << "Q eigenvalue " << i << " = " << eig.eigenvalues()(i) << " should be >= 0";
  }
}

TEST(CtKinematics, LowTurnRateEqualsCV) {
  const double dt = 1.0 / 30.0;
  using Vec5 = Eigen::Matrix<double, 5, 1>;
  Vec5 x;
  x << 10.0, 20.0, 40.0, -15.0, 1e-7;

  auto F = kalman::kinematics::computeF_pixel_CT(x, dt);
  Eigen::Matrix<double, 5, 5> expected = Eigen::Matrix<double, 5, 5>::Identity();
  expected(0, 2) = dt;
  expected(1, 3) = dt;
  EXPECT_TRUE(F.isApprox(expected, 1e-6)) << "F should reduce to the CV transition for w ~ 0";
}

// -----------------------------------------------------------------------
//  Straight-line tracking consistency
// -----------------------------------------------------------------------

TEST(ImmStraightLine, ConsistentTracking) {
  constexpr double dt = 1.0 / 30.0;
  constexpr int steps = 300;
  const auto states = cvStates(steps, dt, 120.0, -60.0);

  Imm imm;
  imm.setTransitionProbabilities(0.02);
  imm.setProcessNoiseSpectralDensities({2000.0, 4000.0, 2500.0});

  Imm::MatrixM R;
  R << 25.0, 0.0, 0.0, 25.0;  // sigma = 5 px
  const auto nees = driveIMM(imm, states, dt, R);

  double mean = 0.0;
  for (double n : nees) mean += n;
  mean /= static_cast<double>(nees.size());

  EXPECT_GT(mean, 0.3) << "Mean NEES = " << mean << " too low (filter overconfident)";
  EXPECT_LT(mean, 12.0) << "Mean NEES = " << mean << " too high (filter inconsistent)";

  const auto err = states.back() - imm.getState();
  for (int i = 0; i < 4; ++i) {
    const double sigma = std::sqrt(imm.getCovariance()(i, i));
    EXPECT_LT(std::abs(err(i)), 3.0 * sigma)
        << "Final state error " << err(i) << " exceeds 3-sigma = " << (3.0 * sigma);
  }
}

// -----------------------------------------------------------------------
//  Maneuver: IMM beats a single CV filter and favours the CT model
// -----------------------------------------------------------------------

TEST(ImmManeuver, BeatsSingleCVAndUsesCT) {
  constexpr double dt = 1.0 / 30.0;
  constexpr int steps = 150;
  constexpr double v = 200.0;  // px/s
  constexpr double w = 0.6;    // rad/s -> R ~ 333 px, ~2.0 rad total turn
  const auto states = ctStates(steps, dt, v, w);
  constexpr double sigma = 5.0;

  // Same noise draws for both filters (identical seed).
  std::mt19937 rng(7);
  std::normal_distribution<double> noise(0.0, sigma);
  std::vector<Imm::VectorM> zs;
  zs.reserve(steps);
  for (int k = 0; k < steps; ++k) {
    Imm::VectorM z = states[k].template head<2>();
    z(0) += noise(rng);
    z(1) += noise(rng);
    zs.push_back(z);
  }

  Imm::MatrixC P0 = Imm::MatrixC::Identity() * (80.0 * 80.0);
  P0(2, 2) = 60.0 * 60.0;
  P0(3, 3) = 60.0 * 60.0;
  Imm::MatrixM R = Imm::MatrixM::Identity() * (sigma * sigma);

  Imm imm;
  imm.setTransitionProbabilities(0.02);
  imm.setProcessNoiseSpectralDensities({2000.0, 4000.0, 2500.0});
  imm.setTurnRatePrior(w);
  imm.initialize(states[0], P0);
  double imm_vel = 0.0;
  for (int k = 1; k < steps; ++k) {
    imm.predict(dt);
    imm.update(zs[k], R, 9.21);
    imm_vel += (states[k].tail<2>() - imm.getState().tail<2>()).norm();
  }
  imm_vel /= (steps - 1);

  PixelCvFilter cv;
  cv.setProcessNoiseSpectralDensity(2000.0);
  cv.setMeasurementNoiseCovariance(R);
  cv.setState(states[0]);
  cv.setCovariance(P0);
  double cv_vel = 0.0;
  for (int k = 1; k < steps; ++k) {
    cv.predict(dt);
    cv.update(zs[k]);
    cv_vel += (states[k].tail<2>() - cv.getState().tail<2>()).norm();
  }
  cv_vel /= (steps - 1);

  // On a sustained turn the CT model's velocity estimate must beat a plain CV
  // filter (which lags the turn).  Position errors are dominated by
  // measurement noise, so the velocity error is the discriminating metric.
  EXPECT_LT(imm_vel, 0.8 * cv_vel)
      << "IMM mean velocity error " << imm_vel << " vs single-CV " << cv_vel;

  const auto mu = imm.modeProbabilities();
  EXPECT_GT(mu(2), mu(0)) << "CT mode probability " << mu(2) << " should exceed CV (" << mu(0)
                          << ") on a turning target";
  EXPECT_GT(mu(2), mu(1)) << "CT mode probability " << mu(2) << " should exceed CA (" << mu(1)
                          << ") on a turning target";
}

// -----------------------------------------------------------------------
//  Outlier gating
// -----------------------------------------------------------------------

TEST(ImmGating, RejectsOutlierWithoutChangingState) {
  constexpr double dt = 1.0 / 30.0;
  const auto states = cvStates(3, dt, 60.0, 40.0);

  Imm imm;
  imm.setProcessNoiseSpectralDensities({2000.0, 4000.0, 2500.0});
  Imm::MatrixC P0 = Imm::MatrixC::Identity() * 100.0;
  imm.initialize(states[0], P0);
  imm.predict(dt);

  const Imm::VectorC before = imm.getState();
  Imm::VectorM z;
  z << 10000.0, 10000.0;  // far outlier
  const bool accepted = imm.update(z, Imm::MatrixM::Identity() * 25.0, 9.21);

  EXPECT_FALSE(accepted) << "Far-outlier measurement should be gated out";
  EXPECT_TRUE(imm.getState().isApprox(before)) << "Gated update must not modify the state";
}

// -----------------------------------------------------------------------
//  Adaptive measurement noise
// -----------------------------------------------------------------------

TEST(ImmAdaptiveNoise, ShrinksScaleWhenInnovationsArePersistentlySmall) {
  constexpr double dt = 1.0 / 30.0;
  constexpr int steps = 400;
  const auto states = cvStates(steps, dt, 100.0, 50.0);

  Imm imm;
  imm.setAdaptiveNoise(true, 30, 2.0);
  imm.setProcessNoiseSpectralDensities({2000.0, 4000.0, 2500.0});

  // Noise-free measurements on a straight line: the model over-predicts the
  // innovation (reported R = sigma 20 px), so per-step innovations are
  // persistently near zero and the scale must shrink (the same regime the app
  // saw with NEES ~ 0.09).
  Imm::MatrixM R;
  R << 400.0, 0.0, 0.0, 400.0;
  driveIMM(imm, states, dt, R, 9.21, 7, 0.0);

  EXPECT_LT(imm.noiseScale(), 0.6)
      << "Adaptive noise scale = " << imm.noiseScale() << " should shrink below 1";
}

TEST(ImmAdaptiveNoise, StaysAtOneWhenDisabled) {
  constexpr double dt = 1.0 / 30.0;
  constexpr int steps = 400;
  const auto states = cvStates(steps, dt, 100.0, 50.0);

  Imm imm;
  imm.setAdaptiveNoise(false);
  imm.setProcessNoiseSpectralDensities({2000.0, 4000.0, 2500.0});
  Imm::MatrixM R;
  R << 400.0, 0.0, 0.0, 400.0;
  driveIMM(imm, states, dt, R);

  EXPECT_DOUBLE_EQ(imm.noiseScale(), 1.0);
}