#pragma once

#include <gtest/gtest.h>

#include <Eigen/Core>
#include <cmath>
#include <random>
#include <vector>

#include "kalman/core/filter.hpp"

namespace helpers {

/// Verify F(0) == I and Q is symmetric positive-definite.
template <typename Filter>
void testProcessModelStructure(double qc) {
  using T = typename Filter::Scalar;
  using Policy = typename Filter::PolicyType;
  constexpr int N = Filter::StateDim;
  using ModelTag = typename Policy::ModelTag;

  auto F0 = kalman::ProcessModel<T, N, ModelTag>::F(T(0));
  ASSERT_EQ(F0.rows(), N);
  ASSERT_EQ(F0.cols(), N);
  EXPECT_TRUE(F0.isIdentity()) << "F(0) should be identity";

  auto F_dt = kalman::ProcessModel<T, N, ModelTag>::F(T(0.1));
  EXPECT_FALSE(F_dt.isIdentity()) << "F(dt>0) should not be identity";

  auto Q = kalman::ProcessModel<T, N, ModelTag>::Q(T(0.1), T(qc));
  ASSERT_EQ(Q.rows(), N);
  ASSERT_EQ(Q.cols(), N);
  EXPECT_TRUE(Q.isApprox(Q.transpose(), T(1e-10))) << "Q should be symmetric";

  Eigen::SelfAdjointEigenSolver<Eigen::Matrix<T, N, N>> eig(Q);
  for (int i = 0; i < N; ++i) {
    EXPECT_GE(eig.eigenvalues()(i), T(-1e-6))
        << "Q eigenvalue " << i << " = " << eig.eigenvalues()(i) << " should be >= 0";
  }
}

/// Check numerical stability with small, large, negative, and zero dt.
template <typename Filter>
void testNumericalStability(double qc) {
  using T = typename Filter::Scalar;
  using Policy = typename Filter::PolicyType;
  constexpr int N = Filter::StateDim;
  using ModelTag = typename Policy::ModelTag;

  auto F_small = kalman::ProcessModel<T, N, ModelTag>::F(T(1e-6));
  auto Q_small = kalman::ProcessModel<T, N, ModelTag>::Q(T(1e-6), T(qc));
  EXPECT_TRUE(F_small.allFinite());
  EXPECT_TRUE(Q_small.allFinite());
  EXPECT_TRUE(F_small.isApprox(Eigen::Matrix<T, N, N>::Identity(), T(1e-4)));

  auto F_large = kalman::ProcessModel<T, N, ModelTag>::F(T(1e6));
  auto Q_large = kalman::ProcessModel<T, N, ModelTag>::Q(T(1e6), T(qc));
  EXPECT_TRUE(F_large.allFinite());
  EXPECT_TRUE(Q_large.allFinite());

  auto F_neg = kalman::ProcessModel<T, N, ModelTag>::F(T(-0.1));
  auto Q_neg = kalman::ProcessModel<T, N, ModelTag>::Q(T(-0.1), T(qc));
  EXPECT_TRUE(F_neg.allFinite());
  EXPECT_TRUE(Q_neg.allFinite());

  auto F0 = kalman::ProcessModel<T, N, ModelTag>::F(T(0));
  EXPECT_TRUE(F0.isIdentity());
}

/// Run the filter over a trajectory and check that the Average NEES is
/// below the given chi2 threshold.
template <typename Filter>
void testConvergence(const std::vector<typename Filter::VectorN>& states, double qc, double dt,
                     double sigma_meas, double chi2_threshold, double init_pos_sigma = 100.0,
                     double init_vel_sigma = 50.0, unsigned seed = 42) {
  using T = typename Filter::Scalar;
  using Policy = typename Filter::PolicyType;
  constexpr int N = Filter::StateDim;
  constexpr int M = Filter::MeasDim;

  int n_steps = static_cast<int>(states.size());
  ASSERT_GT(n_steps, 10);

  Policy policy;
  std::mt19937 rng(seed);
  std::normal_distribution<T> noise(T(0), T(sigma_meas));

  std::vector<typename Filter::VectorM> measurements;
  measurements.reserve(n_steps);
  for (const auto& x : states) {
    auto z = policy.h(x);
    for (int i = 0; i < M; ++i) {
      z(i) += noise(rng);
    }
    measurements.push_back(z);
  }

  Filter filter;
  filter.setProcessNoiseSpectralDensity(T(qc));
  auto R = Eigen::Matrix<T, M, M>::Identity() * T(sigma_meas * sigma_meas);
  filter.setMeasurementNoiseCovariance(R);

  filter.setState(states[0]);
  Eigen::Matrix<T, N, N> P0 = Eigen::Matrix<T, N, N>::Identity();
  if constexpr (N >= 2) {
    P0(0, 0) = init_pos_sigma * init_pos_sigma;
    P0(1, 1) = init_pos_sigma * init_pos_sigma;
  }
  if constexpr (N >= 4) {
    P0(2, 2) = init_vel_sigma * init_vel_sigma;
    P0(3, 3) = init_vel_sigma * init_vel_sigma;
  }
  if constexpr (N >= 6) {
    P0(4, 4) = init_vel_sigma * init_vel_sigma;
    P0(5, 5) = init_vel_sigma * init_vel_sigma;
  }
  filter.setCovariance(P0);

  double total_nees = 0.0;
  int nees_count = 0;

  for (int k = 0; k < n_steps; ++k) {
    if (k > 0) {
      filter.filter(T(dt), measurements[k]);
    }

    auto err = states[k] - filter.getState();
    auto P = filter.getCovariance();
    double nees = P.ldlt().solve(err).dot(err);
    total_nees += nees;
    nees_count++;
  }

  double anees = total_nees / nees_count;
  EXPECT_LT(anees, chi2_threshold)
      << "Average NEES = " << anees << " exceeds chi2 threshold = " << chi2_threshold;

  auto final_state = filter.getState();
  auto final_cov = filter.getCovariance();
  for (int i = 0; i < N; ++i) {
    double err = std::abs(final_state(i) - states.back()(i));
    double sigma = std::sqrt(final_cov(i, i));
    EXPECT_LT(err, 3.0 * sigma) << "State component " << i << " error = " << err
                                << " exceeds 3-sigma = " << (3.0 * sigma);
  }
}

/// Monte Carlo NEES consistency test.
template <typename Filter>
void testNEESConsistency(const std::vector<typename Filter::VectorN>& states, double qc, double dt,
                         double sigma_meas, double chi2_low, double chi2_high, int n_runs,
                         double init_pos_sigma = 100.0, double init_vel_sigma = 50.0) {
  using T = typename Filter::Scalar;
  using Policy = typename Filter::PolicyType;
  constexpr int N = Filter::StateDim;
  constexpr int M = Filter::MeasDim;

  int n_steps = static_cast<int>(states.size());
  ASSERT_GT(n_steps, 10);
  ASSERT_GT(n_runs, 5);

  double total_anees = 0.0;

  for (int run = 0; run < n_runs; ++run) {
    Policy policy;
    std::mt19937 rng(1000 + run);
    std::normal_distribution<T> noise(T(0), T(sigma_meas));

    std::vector<typename Filter::VectorM> measurements;
    measurements.reserve(n_steps);
    for (const auto& x : states) {
      auto z = policy.h(x);
      for (int i = 0; i < M; ++i) {
        z(i) += noise(rng);
      }
      measurements.push_back(z);
    }

    Filter filter;
    filter.setProcessNoiseSpectralDensity(T(qc));
    auto R = Eigen::Matrix<T, M, M>::Identity() * T(sigma_meas * sigma_meas);
    filter.setMeasurementNoiseCovariance(R);

    filter.setState(states[0]);
    Eigen::Matrix<T, N, N> P0 = Eigen::Matrix<T, N, N>::Identity();
    if constexpr (N >= 2) {
      P0(0, 0) = init_pos_sigma * init_pos_sigma;
      P0(1, 1) = init_pos_sigma * init_pos_sigma;
    }
    if constexpr (N >= 4) {
      P0(2, 2) = init_vel_sigma * init_vel_sigma;
      P0(3, 3) = init_vel_sigma * init_vel_sigma;
    }
    if constexpr (N >= 6) {
      P0(4, 4) = init_vel_sigma * init_vel_sigma;
      P0(5, 5) = init_vel_sigma * init_vel_sigma;
    }
    filter.setCovariance(P0);

    double total_nees = 0.0;
    int nees_count = 0;

    for (int k = 0; k < n_steps; ++k) {
      if (k > 0) {
        filter.filter(T(dt), measurements[k]);
      }

      auto err = states[k] - filter.getState();
      auto P = filter.getCovariance();
      double nees = P.ldlt().solve(err).dot(err);
      total_nees += nees;
      nees_count++;
    }

    total_anees += total_nees / nees_count;
  }

  double mean_anees = total_anees / n_runs;
  EXPECT_GT(mean_anees, chi2_low) << "Mean ANEES = " << mean_anees
                                  << " below lower bound = " << chi2_low;
  EXPECT_LT(mean_anees, chi2_high)
      << "Mean ANEES = " << mean_anees << " above upper bound = " << chi2_high;
}

}  // namespace helpers
