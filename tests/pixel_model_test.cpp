/// @file
/// @brief Unit tests for the 2-D pixel-space UKF models (CV and CA).
///
/// Covers:
///   - Measurement model correctness
///   - Process model structure and symmetry / positive-definiteness of Q
///   - Numerical stability at extreme time steps
///   - Full filter convergence on a figure-8 trajectory (NEES & 3σ bounds)
///   - Monte Carlo NEES consistency
///
/// Uses Google Test (gtest).

#include <gtest/gtest.h>

#include <Eigen/Cholesky>
#include <Eigen/Core>
#include <Eigen/Eigenvalues>
#include <numbers>
#include <random>

#include "kalman/pixel_measurement_model.hpp"
#include "kalman/unscented_kalman_filter.hpp"

namespace {

using kalman::PixelCAPolicy;
using kalman::PixelPolicy;
using kalman::UnscentedKalmanFilter;

constexpr double kEps = 10.0 * Eigen::NumTraits<double>::epsilon();

// ---------------------------------------------------------------------------
//  4-state pixel CV model tests
// ---------------------------------------------------------------------------

using PixelFilter = UnscentedKalmanFilter<double, 4, 2, PixelPolicy>;

/// h([u, v, vu, vv]) should extract [u, v].
TEST(PixelMeasurementTest, ReturnsUV) {
  Eigen::Matrix<double, 4, 1> x;
  x << 1920.0, 1080.0, 100.0, 50.0;
  auto z = PixelPolicy::h(x);
  EXPECT_NEAR(z(0), 1920.0, kEps);
  EXPECT_NEAR(z(1), 1080.0, kEps);
}

/// Validate the sparsity and values of the 4-state pixel CV transition matrix.
TEST(PixelProcessModelTest, FStructure) {
  double dt = 0.1;
  auto F = kalman::computeF_pixel(dt);

  // Identity diagonal
  for (int i = 0; i < 4; ++i) {
    EXPECT_NEAR(F(i, i), 1.0, kEps);
  }

  // Position ← Velocity
  EXPECT_NEAR(F(0, 2), dt, kEps);
  EXPECT_NEAR(F(1, 3), dt, kEps);

  // Exactly 6 non-zeros (4 diag + 2 off-diag)
  int non_zero = 0;
  for (int i = 0; i < 4; ++i) {
    for (int j = 0; j < 4; ++j) {
      if (std::abs(F(i, j)) > kEps) ++non_zero;
    }
  }
  EXPECT_EQ(non_zero, 6);
}

/// The pixel-space CV process-noise covariance must be symmetric and PSD.
TEST(PixelProcessModelTest, QSymmetricAndPSD) {
  double dt = 0.01;
  double qc = 100.0;
  auto Q = kalman::computeQ_pixel(dt, qc);

  // Symmetry
  EXPECT_TRUE(Q.isApprox(Q.transpose(), kEps));

  // Positive semi-definite
  Eigen::SelfAdjointEigenSolver<Eigen::Matrix<double, 4, 4>> solver(Q);
  for (int i = 0; i < 4; ++i) {
    EXPECT_GE(solver.eigenvalues()(i), -kEps);
  }
}

TEST(PixelNumericalStabilityTest, SmallDt) {
  PixelFilter filter;
  filter.setState(PixelFilter::VectorN::Ones());
  filter.setCovariance(PixelFilter::MatrixN::Identity());

  filter.predict(1e-6);
  EXPECT_TRUE(filter.getState().allFinite());
  EXPECT_TRUE(filter.getCovariance().allFinite());
}

TEST(PixelNumericalStabilityTest, LargeDt) {
  PixelFilter filter;
  filter.setState(PixelFilter::VectorN::Ones());
  filter.setCovariance(PixelFilter::MatrixN::Identity());

  filter.predict(10.0);
  EXPECT_TRUE(filter.getState().allFinite());
  EXPECT_TRUE(filter.getCovariance().allFinite());
}

TEST(PixelNumericalStabilityTest, NegativeDtNoOp) {
  PixelFilter filter;
  filter.setState(PixelFilter::VectorN::Ones() * 42.0);
  filter.setCovariance(PixelFilter::MatrixN::Identity() * 10.0);

  auto x_before = filter.getState();
  auto P_before = filter.getCovariance();

  filter.predict(-1.0);

  EXPECT_EQ(filter.getState(), x_before);
  EXPECT_EQ(filter.getCovariance(), P_before);
}

// ---------------------------------------------------------------------------
//  Pixel-space convergence test — figure-8 Lissajous
// ---------------------------------------------------------------------------

/// A 4-D pixel-space trajectory sample.
struct PixelTrajectorySample {
  double t;
  Eigen::Matrix<double, 4, 1> x;  // [u, v, vu, vv]
  Eigen::Matrix<double, 2, 1> z;  // [u, v] (noise-free measurement)
};

/// Generate a figure-8 Lissajous trajectory in pixel coordinates.
std::vector<PixelTrajectorySample> generatePixelTrajectory(double total_time, double dt) {
  const double cx = 1920.0, cy = 1080.0;
  const double A_u = 1300.0, A_v = 600.0;
  const double omega = 0.4;

  int n = static_cast<int>(std::round(total_time / dt));
  std::vector<PixelTrajectorySample> traj;
  traj.reserve(n);

  for (int i = 0; i < n; ++i) {
    double t = i * dt;
    PixelTrajectorySample s{};
    s.t = t;
    double u = cx + A_u * std::cos(omega * t);
    double v = cy + A_v * std::sin(2.0 * omega * t);
    double vu = -A_u * omega * std::sin(omega * t);
    double vv = 2.0 * A_v * omega * std::cos(2.0 * omega * t);
    s.x << u, v, vu, vv;
    s.z = PixelPolicy::h(s.x);
    traj.push_back(s);
  }
  return traj;
}

/// Full convergence test for the 4-state pixel CV filter on a figure-8
/// trajectory.  Validates NEES < χ²(4) and final position within 3σ.
TEST(PixelConvergenceTest, Figure8Trajectory) {
  const double total_time = 10.0;
  const double dt = 0.01;
  const double sigma_uv = 2.0;
  const double qc = 10000.0;

  auto traj = generatePixelTrajectory(total_time, dt);

  PixelFilter filter(1.0, 2.0, 0.0);
  filter.setProcessNoiseSpectralDensity(qc);

  PixelFilter::MatrixM R;
  R << sigma_uv * sigma_uv, 0.0, 0.0, sigma_uv * sigma_uv;
  filter.setMeasurementNoiseCovariance(R);

  // Initialise close to the true state at t=0, with moderate uncertainty
  const auto& s0 = traj[0];
  PixelFilter::VectorN x_init;
  x_init << s0.x(0) + 50.0, s0.x(1) - 30.0, s0.x(2), s0.x(3) + 40.0;
  filter.setState(x_init);

  PixelFilter::MatrixN P0 = PixelFilter::MatrixN::Identity();
  P0(0, 0) = 10000.0;
  P0(1, 1) = 10000.0;
  P0(2, 2) = 10000.0;
  P0(3, 3) = 10000.0;
  filter.setCovariance(P0);

  std::mt19937 rng(42);
  std::normal_distribution<double> noise_u(0.0, sigma_uv);
  std::normal_distribution<double> noise_v(0.0, sigma_uv);

  std::vector<double> nees_history;
  nees_history.reserve(traj.size());

  for (size_t i = 0; i < traj.size(); ++i) {
    const auto& s = traj[i];

    PixelFilter::VectorM z;
    z(0) = s.z(0) + noise_u(rng);
    z(1) = s.z(1) + noise_v(rng);

    if (i > 0) {
      filter.filter(dt, z);
    }

    auto err = s.x - filter.getState();
    auto P = filter.getCovariance();
    double nees = P.ldlt().solve(err).dot(err);
    nees_history.push_back(nees);
  }

  const double chi2_99 = 13.277;  // 4 DOF, 99th percentile
  int half = static_cast<int>(nees_history.size() / 2);
  for (size_t i = half; i < nees_history.size(); ++i) {
    EXPECT_LT(nees_history[i], chi2_99);
  }

  const auto& final_true = traj.back().x;
  auto final_err = final_true - filter.getState();
  double pos_err = std::sqrt(final_err(0) * final_err(0) + final_err(1) * final_err(1));
  auto P_final = filter.getCovariance();
  double pos_var = P_final(0, 0) + P_final(1, 1);
  double pos_3sigma = 3.0 * std::sqrt(pos_var);
  EXPECT_LT(pos_err, pos_3sigma);
}

// ---------------------------------------------------------------------------
//  Pixel-space NEES consistency test (Monte Carlo)
// ---------------------------------------------------------------------------

/// Monte Carlo NEES consistency for the 4-state pixel CV filter.
TEST(PixelNEESTest, MonteCarloConsistency) {
  const double total_time = 10.0;
  const double dt = 0.01;
  const double sigma_uv = 2.0;
  const double qc = 8000.0;
  const int n_r = 50;

  auto traj = generatePixelTrajectory(total_time, dt);

  const auto& s0 = traj[0];

  PixelFilter::MatrixN P0 = PixelFilter::MatrixN::Identity();
  P0(0, 0) = 10000.0;
  P0(1, 1) = 10000.0;
  P0(2, 2) = 10000.0;
  P0(3, 3) = 10000.0;

  std::vector<double> nees_final;
  nees_final.reserve(n_r);

  for (int run = 0; run < n_r; ++run) {
    PixelFilter filter(1.0, 2.0, 0.0);
    filter.setProcessNoiseSpectralDensity(qc);

    PixelFilter::MatrixM R;
    R << sigma_uv * sigma_uv, 0.0, 0.0, sigma_uv * sigma_uv;
    filter.setMeasurementNoiseCovariance(R);

    PixelFilter::VectorN x_init;
    x_init << s0.x(0) + 50.0, s0.x(1) - 30.0, s0.x(2) + 20.0, s0.x(3) + 40.0;
    filter.setState(x_init);
    filter.setCovariance(P0);

    std::mt19937 rng(1000 + run);
    std::normal_distribution<double> noise_u(0.0, sigma_uv);
    std::normal_distribution<double> noise_v(0.0, sigma_uv);

    for (size_t i = 0; i < traj.size(); ++i) {
      const auto& s = traj[i];

      PixelFilter::VectorM z;
      z(0) = s.z(0) + noise_u(rng);
      z(1) = s.z(1) + noise_v(rng);

      if (i > 0) {
        filter.filter(dt, z);
      }
    }

    auto err = traj.back().x - filter.getState();
    auto P = filter.getCovariance();
    double nees = P.ldlt().solve(err).dot(err);
    nees_final.push_back(nees);
  }

  double anees = std::accumulate(nees_final.begin(), nees_final.end(), 0.0) / n_r;

  double Nnr = 4.0 * n_r;
  double z975 = 1.96;
  double lower = (Nnr - z975 * std::sqrt(2.0 * Nnr)) / n_r;
  double upper = (Nnr + z975 * std::sqrt(2.0 * Nnr)) / n_r;

  EXPECT_GE(anees, lower);
  EXPECT_LE(anees, upper);
}

// ---------------------------------------------------------------------------
//  Pixel CA model tests (6-state: u, v, vu, vv, au, av)
// ---------------------------------------------------------------------------

using PixelCAFilter = UnscentedKalmanFilter<double, 6, 2, PixelCAPolicy>;

/// A 6-D pixel CA trajectory sample.
struct PixelCATrajectorySample {
  double t;
  Eigen::Matrix<double, 6, 1> x;
  Eigen::Matrix<double, 2, 1> z;
};

/// Generate a figure-8 Lissajous in pixel space with analytic acceleration.
std::vector<PixelCATrajectorySample> generatePixelCATrajectory(double total_time, double dt) {
  const double cx = 1920.0, cy = 1080.0;
  const double A_u = 1300.0, A_v = 600.0;
  const double omega = 0.4;

  int n = static_cast<int>(std::round(total_time / dt));
  std::vector<PixelCATrajectorySample> traj;
  traj.reserve(n);

  for (int i = 0; i < n; ++i) {
    double t = i * dt;
    PixelCATrajectorySample s{};
    s.t = t;
    double ct = std::cos(omega * t);
    double st = std::sin(omega * t);
    double u = cx + A_u * ct;
    double v = cy + A_v * std::sin(2.0 * omega * t);
    double vu = -A_u * omega * st;
    double vv = 2.0 * A_v * omega * std::cos(2.0 * omega * t);
    double au = -A_u * omega * omega * ct;
    double av = -4.0 * A_v * omega * omega * std::sin(2.0 * omega * t);
    s.x << u, v, vu, vv, au, av;
    s.z = PixelCAPolicy::h(s.x);
    traj.push_back(s);
  }
  return traj;
}

/// Validate sparsity and values of the 6-D pixel CA transition matrix.
TEST(PixelCAProcessModelTest, FStructure) {
  double dt = 0.1;
  auto F = kalman::computeF_pixel_CA(dt);

  for (int i = 0; i < 6; ++i) EXPECT_NEAR(F(i, i), 1.0, kEps);

  EXPECT_NEAR(F(0, 2), dt, kEps);
  EXPECT_NEAR(F(1, 3), dt, kEps);
  EXPECT_NEAR(F(0, 4), 0.5 * dt * dt, kEps);
  EXPECT_NEAR(F(1, 5), 0.5 * dt * dt, kEps);
  EXPECT_NEAR(F(2, 4), dt, kEps);
  EXPECT_NEAR(F(3, 5), dt, kEps);

  int non_zero = 0;
  for (int i = 0; i < 6; ++i)
    for (int j = 0; j < 6; ++j)
      if (std::abs(F(i, j)) > kEps) ++non_zero;
  EXPECT_EQ(non_zero, 12);
}

/// Pixel CA process-noise covariance must be symmetric and PSD.
TEST(PixelCAProcessModelTest, QSymmetricAndPSD) {
  double dt = 0.01;
  double qc = 1000.0;
  auto Q = kalman::computeQ_pixel_CA(dt, qc);

  EXPECT_TRUE(Q.isApprox(Q.transpose(), kEps));

  Eigen::SelfAdjointEigenSolver<Eigen::Matrix<double, 6, 6>> solver(Q);
  for (int i = 0; i < 6; ++i) EXPECT_GE(solver.eigenvalues()(i), -kEps);
}

TEST(PixelCANumericalStabilityTest, SmallDt) {
  PixelCAFilter filter;
  filter.setState(PixelCAFilter::VectorN::Ones());
  filter.setCovariance(PixelCAFilter::MatrixN::Identity());

  filter.predict(1e-6);
  EXPECT_TRUE(filter.getState().allFinite());
  EXPECT_TRUE(filter.getCovariance().allFinite());
}

TEST(PixelCANumericalStabilityTest, LargeDt) {
  PixelCAFilter filter;
  filter.setState(PixelCAFilter::VectorN::Ones());
  filter.setCovariance(PixelCAFilter::MatrixN::Identity());

  filter.predict(10.0);
  EXPECT_TRUE(filter.getState().allFinite());
  EXPECT_TRUE(filter.getCovariance().allFinite());
}

TEST(PixelCANumericalStabilityTest, NegativeDtNoOp) {
  PixelCAFilter filter;
  filter.setState(PixelCAFilter::VectorN::Ones() * 42.0);
  filter.setCovariance(PixelCAFilter::MatrixN::Identity() * 10.0);

  auto x_before = filter.getState();
  auto P_before = filter.getCovariance();

  filter.predict(-1.0);
  EXPECT_EQ(filter.getState(), x_before);
  EXPECT_EQ(filter.getCovariance(), P_before);
}

/// Full convergence test for the 6-state pixel CA filter on a figure-8.
TEST(PixelCAConvergenceTest, Figure8Trajectory) {
  const double total_time = 10.0;
  const double dt = 0.10;
  const double sigma_uv = 2.0;
  const double qc = 20000.0;

  auto traj = generatePixelCATrajectory(total_time, dt);

  PixelCAFilter filter(1.0, 2.0, 0.0);
  filter.setProcessNoiseSpectralDensity(qc);

  PixelCAFilter::MatrixM R;
  R << sigma_uv * sigma_uv, 0.0, 0.0, sigma_uv * sigma_uv;
  filter.setMeasurementNoiseCovariance(R);

  const auto& s0 = traj[0];
  PixelCAFilter::VectorN x_init;
  x_init << s0.x(0) + 50.0, s0.x(1) - 30.0, s0.x(2), s0.x(3) + 40.0, s0.x(4), s0.x(5);
  filter.setState(x_init);

  PixelCAFilter::MatrixN P0 = PixelCAFilter::MatrixN::Identity();
  P0(0, 0) = 50000.0;
  P0(1, 1) = 50000.0;
  P0(2, 2) = 50000.0;
  P0(3, 3) = 50000.0;
  P0(4, 4) = 50000.0;
  P0(5, 5) = 50000.0;
  filter.setCovariance(P0);

  std::mt19937 rng(42);
  std::normal_distribution<double> noise_u(0.0, sigma_uv);
  std::normal_distribution<double> noise_v(0.0, sigma_uv);

  std::vector<double> nees_history;
  nees_history.reserve(traj.size());

  for (size_t i = 0; i < traj.size(); ++i) {
    const auto& s = traj[i];

    PixelCAFilter::VectorM z;
    z(0) = s.z(0) + noise_u(rng);
    z(1) = s.z(1) + noise_v(rng);

    if (i > 0) filter.filter(dt, z);

    auto err = s.x - filter.getState();
    auto P = filter.getCovariance();
    double nees = P.ldlt().solve(err).dot(err);
    nees_history.push_back(nees);
  }

  const double chi2_99 = 16.812;  // 6 DOF, 99th percentile
  int half = static_cast<int>(nees_history.size() / 2);
  for (size_t i = half; i < nees_history.size(); ++i) EXPECT_LT(nees_history[i], chi2_99);

  const auto& final_true = traj.back().x;
  auto final_err = final_true - filter.getState();
  double pos_err = std::sqrt(final_err(0) * final_err(0) + final_err(1) * final_err(1));
  auto P_final = filter.getCovariance();
  double pos_var = P_final(0, 0) + P_final(1, 1);
  double pos_3sigma = 3.0 * std::sqrt(pos_var);
  EXPECT_LT(pos_err, pos_3sigma);
}

/// Monte Carlo NEES consistency for the 6-state pixel CA filter.
TEST(PixelCANEESTest, MonteCarloConsistency) {
  const double total_time = 10.0;
  const double dt = 0.10;
  const double sigma_uv = 2.0;
  const double qc = 2500.0;
  const int n_r = 30;

  auto traj = generatePixelCATrajectory(total_time, dt);

  const auto& s0 = traj[0];

  PixelCAFilter::MatrixN P0 = PixelCAFilter::MatrixN::Identity();
  P0(0, 0) = 50000.0;
  P0(1, 1) = 50000.0;
  P0(2, 2) = 50000.0;
  P0(3, 3) = 50000.0;
  P0(4, 4) = 50000.0;
  P0(5, 5) = 50000.0;

  std::vector<double> nees_final;
  nees_final.reserve(n_r);

  for (int run = 0; run < n_r; ++run) {
    PixelCAFilter filter(1.0, 2.0, 0.0);
    filter.setProcessNoiseSpectralDensity(qc);

    PixelCAFilter::MatrixM R;
    R << sigma_uv * sigma_uv, 0.0, 0.0, sigma_uv * sigma_uv;
    filter.setMeasurementNoiseCovariance(R);

    PixelCAFilter::VectorN x_init;
    x_init << s0.x(0) + 50.0, s0.x(1) - 30.0, s0.x(2) + 20.0, s0.x(3) + 40.0, s0.x(4), s0.x(5);
    filter.setState(x_init);
    filter.setCovariance(P0);

    std::mt19937 rng(1000 + run);
    std::normal_distribution<double> noise_u(0.0, sigma_uv);
    std::normal_distribution<double> noise_v(0.0, sigma_uv);

    for (size_t i = 0; i < traj.size(); ++i) {
      const auto& s = traj[i];

      PixelCAFilter::VectorM z;
      z(0) = s.z(0) + noise_u(rng);
      z(1) = s.z(1) + noise_v(rng);

      if (i > 0) filter.filter(dt, z);
    }

    auto err = traj.back().x - filter.getState();
    auto P = filter.getCovariance();
    double nees = P.ldlt().solve(err).dot(err);
    nees_final.push_back(nees);
  }

  double anees = std::accumulate(nees_final.begin(), nees_final.end(), 0.0) / n_r;

  double Nnr = 6.0 * n_r;
  double z975 = 1.96;
  double lower = (Nnr - z975 * std::sqrt(2.0 * Nnr)) / n_r;
  double upper = (Nnr + z975 * std::sqrt(2.0 * Nnr)) / n_r;

  EXPECT_GE(anees, lower);
  EXPECT_LE(anees, upper);
}

}  // namespace
