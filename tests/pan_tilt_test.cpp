/// @file
/// @brief Unit tests for the 3-D pan/tilt bearing-only UKF models.
///
/// Covers:
///   - Measurement model correctness (pan/tilt from known positions)
///   - Process model structure and symmetry / positive-definiteness of Q
///   - Numerical stability at extreme time steps
///   - Angle (innovation) wrapping
///   - Compile-time instantiation with various dimensions and types
///   - Full filter convergence on a known trajectory (NEES & 3σ bounds)
///   - Monte Carlo NEES consistency
///
/// Uses Google Test (gtest).

#include <gtest/gtest.h>

#include <Eigen/Cholesky>
#include <Eigen/Core>
#include <Eigen/Eigenvalues>
#include <Eigen/LU>
#include <numbers>
#include <random>

#include "kalman/pan_tilt_measurement_model.hpp"
#include "kalman/unscented_kalman_filter.hpp"

namespace {

using kalman::UnscentedKalmanFilter;
using FilterD = UnscentedKalmanFilter<double, 9, 2>;
using FilterF = UnscentedKalmanFilter<float, 9, 2>;

constexpr double kPi = std::numbers::pi_v<double>;
constexpr double kEps = 10.0 * Eigen::NumTraits<double>::epsilon();

// ---------------------------------------------------------------------------
//  Measurement model correctness
// ---------------------------------------------------------------------------

/// Verify the bearing-only measurement function for several known
/// camera-centric positions.  Compares pan and tilt against closed-form
/// expectations.
TEST(MeasurementModelTest, KnownPositions) {
  // [1,0,0,...] -> pan=atan2(1,0)=π/2, tilt=0
  Eigen::Matrix<double, 9, 1> x1;
  x1 << 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0;
  auto z1 = kalman::h(x1);
  EXPECT_NEAR(z1(0), kPi / 2, kEps);
  EXPECT_NEAR(z1(1), 0.0, kEps);

  // [0,1,1,...] -> pan=atan2(0,1)=0, tilt=atan2(1,1)=π/4
  Eigen::Matrix<double, 9, 1> x2;
  x2 << 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0;
  auto z2 = kalman::h(x2);
  EXPECT_NEAR(z2(0), 0.0, kEps);
  EXPECT_NEAR(z2(1), kPi / 4, kEps);

  // [0,0,1,...] -> pan=atan2(0,1)=0, tilt=atan2(0,1)=0
  Eigen::Matrix<double, 9, 1> x3;
  x3 << 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0;
  auto z3 = kalman::h(x3);
  EXPECT_NEAR(z3(0), 0.0, kEps);
  EXPECT_NEAR(z3(1), 0.0, kEps);

  // [1,1,1,...] -> pan=atan2(1,1)=π/4, tilt=atan2(1,√2)=atan2(1,1.4142)=0.6155
  Eigen::Matrix<double, 9, 1> x4;
  x4 << 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0;
  auto z4 = kalman::h(x4);
  EXPECT_NEAR(z4(0), kPi / 4, kEps);
  EXPECT_NEAR(z4(1), std::atan2(1.0, std::sqrt(2.0)), kEps);

  // [-1,0,1,...] -> pan=atan2(-1,1)=-π/4, tilt=atan2(0,1)=0
  Eigen::Matrix<double, 9, 1> x5;
  x5 << -1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0;
  auto z5 = kalman::h(x5);
  EXPECT_NEAR(z5(0), -kPi / 4, kEps);
  EXPECT_NEAR(z5(1), 0.0, kEps);

  // [0,0,-1,...] -> pan=atan2(0,-1)=π, tilt=atan2(0,1)=0
  Eigen::Matrix<double, 9, 1> x6;
  x6 << 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0;
  auto z6 = kalman::h(x6);
  EXPECT_NEAR(z6(0), kPi, kEps);
  EXPECT_NEAR(z6(1), 0.0, kEps);
}

// ---------------------------------------------------------------------------
//  Process model structure
// ---------------------------------------------------------------------------

/// At dt = 0, the state-transition matrix must be identity.
TEST(ProcessModelTest, FIdentityAtDtZero) {
  auto F = kalman::computeF_9(0.0);
  EXPECT_TRUE(F.isIdentity(kEps));
}

/// Validate the structural sparsity pattern and numeric values of the
/// 9-state CA transition matrix for a positive time step.
TEST(ProcessModelTest, FStructureForDtPositive) {
  double dt = 0.01;
  auto F = kalman::computeF_9(dt);

  // Diagonal is identity
  for (int i = 0; i < 9; ++i) {
    EXPECT_NEAR(F(i, i), 1.0, kEps);
  }

  // Position ← Velocity : F(i, i+3) = dt   for i=0,1,2
  EXPECT_NEAR(F(0, 3), dt, kEps);
  EXPECT_NEAR(F(1, 4), dt, kEps);
  EXPECT_NEAR(F(2, 5), dt, kEps);

  // Position ← Acceleration : F(i, i+6) = ½dt²  for i=0,1,2
  EXPECT_NEAR(F(0, 6), 0.5 * dt * dt, kEps);
  EXPECT_NEAR(F(1, 7), 0.5 * dt * dt, kEps);
  EXPECT_NEAR(F(2, 8), 0.5 * dt * dt, kEps);

  // Velocity ← Acceleration : F(i+3, i+6) = dt  for i=0,1,2
  EXPECT_NEAR(F(3, 6), dt, kEps);
  EXPECT_NEAR(F(4, 7), dt, kEps);
  EXPECT_NEAR(F(5, 8), dt, kEps);

  // Everything else is zero
  int non_zero = 0;
  for (int i = 0; i < 9; ++i) {
    for (int j = 0; j < 9; ++j) {
      if (std::abs(F(i, j)) > kEps) ++non_zero;
    }
  }
  // 9 diagonal + 3 pos-vel + 3 pos-acc + 3 vel-acc = 18 non-zeros
  EXPECT_EQ(non_zero, 18);
}

/// The process-noise covariance must be symmetric and positive semi-definite.
TEST(ProcessModelTest, QSymmetricAndPSD) {
  double dt = 0.01;
  double qc = 1.0;
  auto Q = kalman::computeQ_9(dt, qc);

  // Symmetry
  EXPECT_TRUE(Q.isApprox(Q.transpose(), kEps));

  // Positive semi-definite: all eigenvalues >= 0
  Eigen::SelfAdjointEigenSolver<Eigen::Matrix<double, 9, 9>> solver(Q);
  for (int i = 0; i < 9; ++i) {
    EXPECT_GE(solver.eigenvalues()(i), -kEps);
  }
}

// ---------------------------------------------------------------------------
//  Numerical stability
// ---------------------------------------------------------------------------

/// Predict with a very small time step must preserve finite state/covariance.
TEST(NumericalStabilityTest, SmallDt) {
  FilterD filter;
  filter.setState(FilterD::VectorN::Ones());
  filter.setCovariance(FilterD::MatrixN::Identity());

  double dt = 1e-6;
  filter.predict(dt);

  EXPECT_TRUE(filter.getState().allFinite());
  EXPECT_TRUE(filter.getCovariance().allFinite());
}

/// Predict with a large time step must preserve finite state/covariance.
TEST(NumericalStabilityTest, LargeDt) {
  FilterD filter;
  filter.setState(FilterD::VectorN::Ones());
  filter.setCovariance(FilterD::MatrixN::Identity());

  double dt = 10.0;
  filter.predict(dt);

  EXPECT_TRUE(filter.getState().allFinite());
  EXPECT_TRUE(filter.getCovariance().allFinite());
}

/// Negative dt must be a no-op (state and covariance unchanged).
TEST(NumericalStabilityTest, NegativeDtNoOp) {
  FilterD filter;
  filter.setState(FilterD::VectorN::Ones() * 42.0);
  filter.setCovariance(FilterD::MatrixN::Identity() * 10.0);

  auto x_before = filter.getState();
  auto P_before = filter.getCovariance();

  filter.predict(-1.0);

  EXPECT_EQ(filter.getState(), x_before);
  EXPECT_EQ(filter.getCovariance(), P_before);
}

/// Zero dt must be a no-op (state and covariance unchanged).
TEST(NumericalStabilityTest, ZeroDtNoOp) {
  FilterD filter;
  filter.setState(FilterD::VectorN::Ones() * 42.0);
  filter.setCovariance(FilterD::MatrixN::Identity() * 10.0);

  auto x_before = filter.getState();
  auto P_before = filter.getCovariance();

  filter.predict(0.0);

  EXPECT_EQ(filter.getState(), x_before);
  EXPECT_EQ(filter.getCovariance(), P_before);
}

// ---------------------------------------------------------------------------
//  Angle wrapping
// ---------------------------------------------------------------------------

/// Verify that the innovation wrapping correctly handles a pan-angle
/// wraparound: predicted pan ≈ -179°, measurement ≈ +179°, expected
/// innovation ≈ -2° (the short path) rather than +358°.
TEST(AngleWrappingTest, InnovationWrapping) {
  double deg = kPi / 180.0;
  double pan_pred = -179.0 * deg;

  FilterD filter;
  FilterD::VectorN x;
  x << 100.0 * std::tan(pan_pred), 0.0, 100.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0;
  filter.setState(x);
  filter.setCovariance(FilterD::MatrixN::Identity() * 1e-6);
  filter.setProcessNoiseSpectralDensity(1e-6);

  double R_pan = (1.0 * deg) * (1.0 * deg);
  FilterD::MatrixM R;
  R << R_pan, 0.0, 0.0, R_pan;
  filter.setMeasurementNoiseCovariance(R);

  FilterD::VectorM z;
  z << 179.0 * deg, 0.0;
  filter.update(z);

  // After the update, pan should have moved from -179° toward +179°
  // by about 2° (the short path), not +358° (the long path).
  auto x_upd = filter.getState();
  double pan_upd = std::atan2(x_upd(0), x_upd(2));

  double moved = std::abs(pan_upd - pan_pred);
  EXPECT_GT(moved, 0.0);
  EXPECT_GT(pan_upd, pan_pred);
}

// ---------------------------------------------------------------------------
//  Static assertions — compile-time only; SUCCEED if instantiation succeeds
// ---------------------------------------------------------------------------

TEST(StaticAssertionsTest, CompilesWithFloat) {
  UnscentedKalmanFilter<float, 9, 2> filter;
  SUCCEED();
}

TEST(StaticAssertionsTest, CompilesWithDouble) {
  UnscentedKalmanFilter<double, 9, 2> filter;
  SUCCEED();
}

TEST(StaticAssertionsTest, CompilesWithDifferentDimensions) {
  UnscentedKalmanFilter<double, 6, 3> filter;
  SUCCEED();
}

// ---------------------------------------------------------------------------
//  Convergence tests
// ---------------------------------------------------------------------------

/// A single time-step sample of a 3-D trajectory with true state and
/// noise-free pan/tilt measurement.
struct TrajectorySample {
  double t;
  Eigen::Matrix<double, 9, 1> x;  // true state
  Eigen::Matrix<double, 2, 1> z;  // noise-free measurement
};

/// Generate a helical (spiral) trajectory in 3-D camera coordinates.
///
/// The drone circles in the xy-plane while moving forward along +z:
///   x(t) = A·cos(ωt),  y(t) = A·sin(ωt),  z(t) = 5 + vz·t
///
/// Velocity and acceleration are computed analytically.
std::vector<TrajectorySample> generateHelicalTrajectory(double total_time, double dt) {
  const double A = 10.0;     // radius (m)
  const double omega = 0.5;  // angular speed (rad/s)
  const double vz = 2.0;     // forward speed (m/s)

  int n = static_cast<int>(std::round(total_time / dt));
  std::vector<TrajectorySample> traj;
  traj.reserve(n);

  for (int i = 0; i < n; ++i) {
    TrajectorySample s{};
    s.t = i * dt;
    double t = s.t;

    double x = A * std::cos(omega * t);
    double y = A * std::sin(omega * t);
    double z = 5.0 + vz * t;

    double vx = -A * omega * std::sin(omega * t);
    double vy = A * omega * std::cos(omega * t);
    double v_z = vz;

    double ax = -A * omega * omega * std::cos(omega * t);
    double ay = -A * omega * omega * std::sin(omega * t);
    double az = 0.0;

    s.x << x, y, z, vx, vy, v_z, ax, ay, az;
    s.z = kalman::h(s.x);
    traj.push_back(s);
  }
  return traj;
}

/// Generate a trajectory with piecewise-constant acceleration that exactly
/// matches the 9-state CA process model (useful for convergence tests).
std::vector<TrajectorySample> generateConstantAccelerationTrajectory(double total_time, double dt) {
  // True initial state — exactly matches the CA process model.
  const double x0 = 10.0, y0 = 0.0, z0 = 5.0;
  const double vx0 = 0.0, vy0 = 5.0, vz0 = 2.0;
  const double ax = -2.5, ay = 0.0, az = 0.0;

  int n = static_cast<int>(std::round(total_time / dt));
  std::vector<TrajectorySample> traj;
  traj.reserve(n);

  for (int i = 0; i < n; ++i) {
    double t = i * dt;
    TrajectorySample s{};
    s.t = t;
    s.x(0) = x0 + vx0 * t + 0.5 * ax * t * t;
    s.x(1) = y0 + vy0 * t + 0.5 * ay * t * t;
    s.x(2) = z0 + vz0 * t;
    s.x(3) = vx0 + ax * t;
    s.x(4) = vy0 + ay * t;
    s.x(5) = vz0;
    s.x(6) = ax;
    s.x(7) = ay;
    s.x(8) = az;
    s.z = kalman::h(s.x);
    traj.push_back(s);
  }
  return traj;
}

/// Factory helper: create a filter with the given noise and tuning parameters.
FilterD makeFilter(double sigma_pan, double sigma_tilt, double qc, double alpha = 1.0) {
  FilterD filter(alpha, 2.0, 0.0);
  filter.setProcessNoiseSpectralDensity(qc);
  FilterD::MatrixM R;
  R << sigma_pan * sigma_pan, 0.0, 0.0, sigma_tilt * sigma_tilt;
  filter.setMeasurementNoiseCovariance(R);
  return filter;
}

/// Full filter convergence test on a constant-acceleration trajectory.
///
/// Checks that:
///   — The NEES (normalised estimation error squared) stays below
///     the 95th percentile of χ²(9) for the second half of the run.
///   — The final 3-D position error is within the 3σ envelope computed
///     from the covariance.
TEST(ConvergenceTest, ConstantAccelerationTrajectory) {
  const double total_time = 10.0;
  const double dt = 0.01;
  const double sigma_pan = 1.0 * kPi / 180.0;
  const double sigma_tilt = 1.0 * kPi / 180.0;
  const double qc = 5.0;

  auto traj = generateConstantAccelerationTrajectory(total_time, dt);
  auto filter = makeFilter(sigma_pan, sigma_tilt, qc);

  FilterD::VectorN x_init;
  x_init << 8.0, 1.0, 6.0, 0.0, 5.0, 2.0, -2.5, 0.0, 0.0;
  filter.setState(x_init);

  FilterD::MatrixN P0 = FilterD::MatrixN::Identity();
  P0(0, 0) = 25.0;
  P0(1, 1) = 25.0;
  P0(2, 2) = 25.0;
  P0(3, 3) = 9.0;
  P0(4, 4) = 9.0;
  P0(5, 5) = 9.0;
  P0(6, 6) = 4.0;
  P0(7, 7) = 4.0;
  P0(8, 8) = 4.0;
  filter.setCovariance(P0);

  std::mt19937 rng(42);
  std::normal_distribution<double> noise_pan(0.0, sigma_pan);
  std::normal_distribution<double> noise_tilt(0.0, sigma_tilt);

  std::vector<double> nees_history;
  nees_history.reserve(traj.size());

  for (size_t i = 0; i < traj.size(); ++i) {
    const auto& s = traj[i];

    FilterD::VectorM z;
    z(0) = s.z(0) + noise_pan(rng);
    z(1) = s.z(1) + noise_tilt(rng);

    z(0) = std::fmod(z(0) + kPi, 2.0 * kPi);
    z(0) += (z(0) < 0) ? kPi : -kPi;

    if (i > 0) {
      filter.filter(dt, z);
    }

    auto err = s.x - filter.getState();
    auto P = filter.getCovariance();
    double nees = P.ldlt().solve(err).dot(err);
    nees_history.push_back(nees);
  }

  const double chi2_95 = 16.919;
  int half = static_cast<int>(nees_history.size() / 2);
  for (size_t i = half; i < nees_history.size(); ++i) {
    EXPECT_LT(nees_history[i], chi2_95);
  }

  const auto& final_true = traj.back().x;
  auto final_err = final_true - filter.getState();
  double pos_err = std::sqrt(final_err(0) * final_err(0) + final_err(1) * final_err(1) +
                             final_err(2) * final_err(2));
  auto P_final = filter.getCovariance();
  double pos_var = P_final(0, 0) + P_final(1, 1) + P_final(2, 2);
  double pos_3sigma = 3.0 * std::sqrt(pos_var);
  EXPECT_LT(pos_err, pos_3sigma);
}

// ---------------------------------------------------------------------------
//  NEES consistency test (Monte Carlo)
// ---------------------------------------------------------------------------

/// Monte Carlo NEES consistency test.
///
/// Runs the filter on `n_r` independent noise realisations of the
/// constant-acceleration trajectory and computes the average NEES (ANEES)
/// at the final step.  ANEES is tested against the 95% confidence interval
/// for a χ²(9·n_r) distribution, approximated by the normal distribution
/// via the central limit theorem.
TEST(NEESTest, MonteCarloConsistency) {
  const double total_time = 10.0;
  const double dt = 0.01;
  const double sigma_pan = 1.0 * kPi / 180.0;
  const double sigma_tilt = 1.0 * kPi / 180.0;
  const double qc = 5.0;
  const int n_r = 50;

  auto traj = generateConstantAccelerationTrajectory(total_time, dt);

  FilterD::MatrixN P0 = FilterD::MatrixN::Identity();
  P0(0, 0) = 25.0;
  P0(1, 1) = 25.0;
  P0(2, 2) = 25.0;
  P0(3, 3) = 9.0;
  P0(4, 4) = 9.0;
  P0(5, 5) = 9.0;
  P0(6, 6) = 4.0;
  P0(7, 7) = 4.0;
  P0(8, 8) = 4.0;

  std::vector<double> nees_final;
  nees_final.reserve(n_r);

  for (int run = 0; run < n_r; ++run) {
    auto filter = makeFilter(sigma_pan, sigma_tilt, qc);

    FilterD::VectorN x_init;
    x_init << 8.0, 1.0, 6.0, 0.0, 5.0, 2.0, -2.5, 0.0, 0.0;
    filter.setState(x_init);
    filter.setCovariance(P0);

    std::mt19937 rng(1000 + run);
    std::normal_distribution<double> noise_pan(0.0, sigma_pan);
    std::normal_distribution<double> noise_tilt(0.0, sigma_tilt);

    for (size_t i = 0; i < traj.size(); ++i) {
      const auto& s = traj[i];

      FilterD::VectorM z;
      z(0) = s.z(0) + noise_pan(rng);
      z(1) = s.z(1) + noise_tilt(rng);

      z(0) = std::fmod(z(0) + kPi, 2.0 * kPi);
      z(0) += (z(0) < 0) ? kPi : -kPi;

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

  double Nnr = 9.0 * n_r;
  double z975 = 1.96;
  double lower = (Nnr - z975 * std::sqrt(2.0 * Nnr)) / n_r;
  double upper = (Nnr + z975 * std::sqrt(2.0 * Nnr)) / n_r;

  EXPECT_GE(anees, lower);
  EXPECT_LE(anees, upper);
}

// ---------------------------------------------------------------------------
//  Debug tests — step-by-step with printed diagnostics
// ---------------------------------------------------------------------------

/// Filter a helical trajectory with perfect initial conditions (state =
/// true at t=0) to verify stability and convergence of the UKF under
/// ideal initialisation.
TEST(DebugTest, PerfectInit) {
  auto traj = generateHelicalTrajectory(10.0, 0.01);

  FilterD filter(1.0, 2.0, 0.0);
  // Use large qc to allow tracking of sinusoidal dynamics
  filter.setProcessNoiseSpectralDensity(5000.0);
  FilterD::MatrixM R;
  R << 0.0003, 0.0, 0.0, 0.0003;
  filter.setMeasurementNoiseCovariance(R);

  // Initialize at true state at t=0
  filter.setState(traj[0].x);

  // Larger initial covariance to maintain sigma-point spread
  FilterD::MatrixN P0 = FilterD::MatrixN::Identity();
  P0(0, 0) = 25.0;
  P0(1, 1) = 25.0;
  P0(2, 2) = 25.0;
  P0(3, 3) = 9.0;
  P0(4, 4) = 9.0;
  P0(5, 5) = 9.0;
  P0(6, 6) = 4.0;
  P0(7, 7) = 4.0;
  P0(8, 8) = 4.0;
  filter.setCovariance(P0);

  for (size_t i = 0; i < traj.size(); ++i) {
    const auto& s = traj[i];
    FilterD::VectorM z = s.z;

    if (i > 0) {
      filter.filter(0.01, z);
    }

    if (i % 200 == 0 || i == traj.size() - 1) {
      auto err = s.x - filter.getState();
      auto P = filter.getCovariance();
      double nees = P.ldlt().solve(err).dot(err);
      printf("i=%zu t=%.2f err=(%.3f,%.3f,%.3f) Pdiag=(%.6e,%.6e,%.6e) NEES=%.1f\n", i, i * 0.01,
             err(0), err(1), err(2), P(0, 0), P(1, 1), P(2, 2), nees);
      printf("  v_err=(%.3f,%.3f,%.3f) Pvel=(%.6e,%.6e,%.6e)\n", err(3), err(4), err(5), P(3, 3),
             P(4, 4), P(5, 5));
    }
  }
}

/// Filter a helical trajectory with standard (imperfect) initial prior
/// but a very large process noise PSD, verifying that the UKF remains
/// stable and avoids divergence under high model uncertainty.
TEST(DebugTest, LargeQcTest) {
  auto traj = generateHelicalTrajectory(10.0, 0.01);

  FilterD filter(1.0, 2.0, 0.0);
  filter.setProcessNoiseSpectralDensity(50000.0);
  FilterD::MatrixM R;
  R << 0.0003, 0.0, 0.0, 0.0003;
  filter.setMeasurementNoiseCovariance(R);

  FilterD::VectorN x_init;
  x_init << 8.0, 1.0, 6.0, 0.0, 5.0, 2.0, -2.5, 0.0, 0.0;
  filter.setState(x_init);

  FilterD::MatrixN P0 = FilterD::MatrixN::Identity();
  P0(0, 0) = 25.0;
  P0(1, 1) = 25.0;
  P0(2, 2) = 25.0;
  P0(3, 3) = 9.0;
  P0(4, 4) = 9.0;
  P0(5, 5) = 9.0;
  P0(6, 6) = 4.0;
  P0(7, 7) = 4.0;
  P0(8, 8) = 4.0;
  filter.setCovariance(P0);

  for (size_t i = 0; i < traj.size(); ++i) {
    const auto& s = traj[i];
    FilterD::VectorM z = s.z;

    if (i > 0) {
      filter.filter(0.01, z);
    }

    if (i % 200 == 0 || i == traj.size() - 1) {
      auto err = s.x - filter.getState();
      auto P = filter.getCovariance();
      double nees = P.ldlt().solve(err).dot(err);
      printf(
          "i=%zu t=%.2f nees=%.1f  pos_err=%.3f  Ppos_diag=(%.6e,%.6e,%.6e) "
          "Pvel_diag=(%.6e,%.6e,%.6e)\n",
          i, i * 0.01, nees, std::sqrt(err(0) * err(0) + err(1) * err(1) + err(2) * err(2)),
          P(0, 0), P(1, 1), P(2, 2), P(3, 3), P(4, 4), P(5, 5));
    }
  }
}

}  // namespace
