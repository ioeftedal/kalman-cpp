/// @file
/// @brief Diagnostic test for the 3-D pan/tilt UKF — prints intermediate values.

#include <Eigen/Cholesky>
#include <Eigen/Core>
#include <cstdio>
#include <iostream>
#include <numbers>
#include <random>

#include "kalman/pan_tilt_measurement_model.hpp"
#include "kalman/unscented_kalman_filter.hpp"

namespace {

using kalman::UnscentedKalmanFilter;
using FilterD = UnscentedKalmanFilter<double, 9, 2>;

constexpr double kPi = std::numbers::pi_v<double>;

struct TrajectorySample {
  double t;
  Eigen::Matrix<double, 9, 1> x;
  Eigen::Matrix<double, 2, 1> z;
};

std::vector<TrajectorySample> generateConstantAccelerationTrajectory(double total_time, double dt) {
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

}  // namespace

int main() {
  const double total_time = 10.0;
  const double dt = 0.01;
  const double sigma_pan = 1.0 * kPi / 180.0;
  const double sigma_tilt = 1.0 * kPi / 180.0;
  const double qc = 5.0;

  auto traj = generateConstantAccelerationTrajectory(total_time, dt);

  FilterD filter(1.0, 2.0, 0.0);
  filter.setProcessNoiseSpectralDensity(qc);

  FilterD::MatrixM R;
  R << sigma_pan * sigma_pan, 0.0, 0.0, sigma_tilt * sigma_tilt;
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

  std::mt19937 rng(42);
  std::normal_distribution<double> noise_pan(0.0, sigma_pan);
  std::normal_distribution<double> noise_tilt(0.0, sigma_tilt);

  printf("Step,Time,NEES,PosErr_x,PosErr_y,PosErr_z,P_xx,P_yy,P_zz,Pan_Innov,Tilt_Innov\n");

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

    // Print first 100 steps then every 100th after that
    if (i < 10 || (i < 100 && i % 10 == 0) || i % 100 == 0 || i == traj.size() - 1) {
      printf("%zu,%.4f,%.3f,%.3f,%.3f,%.3f,%.6e,%.6e,%.6e\n", i, i * dt, nees, err(0), err(1),
             err(2), P(0, 0), P(1, 1), P(2, 2));
    }
  }

  auto final_err = traj.back().x - filter.getState();
  double pos_err = std::sqrt(final_err(0) * final_err(0) + final_err(1) * final_err(1) +
                             final_err(2) * final_err(2));
  auto P_final = filter.getCovariance();
  double pos_var = P_final(0, 0) + P_final(1, 1) + P_final(2, 2);
  double pos_3sigma = 3.0 * std::sqrt(pos_var);

  printf("\nFinal pos_err=%.3f pos_3sigma=%.3f\n", pos_err, pos_3sigma);
  printf("\nFinal state:\n");
  std::cout << filter.getState().transpose() << std::endl;
  printf("\nTrue state:\n");
  std::cout << traj.back().x.transpose() << std::endl;

  return 0;
}
