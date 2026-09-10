/// @file
/// @brief Unscented Kalman Filter (UKF) — general-purpose N-state / M-measurement template.
///
/// The implementation follows Julier & Uhlmann (1997) with the scaled
/// sigma-point parameterisation (Van der Merwe & Wan, 2001).

#pragma once

#include <Eigen/Cholesky>
#include <Eigen/Core>
#include <Eigen/Eigenvalues>
#include <Eigen/LU>
#include <limits>

#include "kalman/core/concepts.hpp"
#include "kalman/core/fwd.hpp"
#include "kalman/core/profiling.hpp"

namespace kalman {

/// Unscented Kalman Filter for state dimension N and measurement dimension M.
///
/// @tparam T      Floating-point scalar type (e.g. `float`, `double`).
/// @tparam N      State vector dimension.
/// @tparam M      Measurement vector dimension.
/// @tparam Policy Policy class providing the measurement model `h(x)`,
///                `wrapInnovation(nu)`, and `ModelTag` for compile-time
///                process model selection.  Must satisfy `KalmanPolicy`.
template <typename T, int N, int M, typename Policy>
class UnscentedKalmanFilter {
  static_assert(N > 0, "State dimension N must be positive");
  static_assert(M > 0, "Measurement dimension M must be positive");
  static_assert(is_kalman_policy_v<Policy>,
                "Policy must satisfy KalmanPolicy (provide ModelTag and "
                "isMeasurementLinear)");

 public:
  using Scalar = T;
  static constexpr int StateDim = N;
  static constexpr int MeasDim = M;

  using PolicyType = Policy;
  using VectorN = Eigen::Matrix<T, N, 1>;
  using MatrixN = Eigen::Matrix<T, N, N>;
  using VectorM = Eigen::Matrix<T, M, 1>;
  using MatrixM = Eigen::Matrix<T, M, M>;
  using MatrixNM = Eigen::Matrix<T, N, M>;
  using MatrixNX = Eigen::Matrix<T, N, 2 * N + 1>;
  using MatrixMX = Eigen::Matrix<T, M, 2 * N + 1>;
  using VectorW = Eigen::Matrix<T, 2 * N + 1, 1>;

  /// Construct with default-constructed policy.
  explicit UnscentedKalmanFilter(T alpha = T(1e-3), T beta = T(2), T kappa = T(0)) noexcept
      : alpha_(alpha), beta_(beta), kappa_(kappa) {
    if (T(N) + kappa_ <= T(0)) {
      kappa_ = T(3) - T(N);
    }
    lambda_ = alpha_ * alpha_ * (T(N) + kappa_) - T(N);
    computeWeights();
    x_.setZero();
    P_.setIdentity();
    qc_ = T(1);
    R_.setIdentity();
  }

  /// Construct with a custom policy instance (e.g. for stateful policies).
  explicit UnscentedKalmanFilter(Policy policy, T alpha = T(1e-3), T beta = T(2),
                                 T kappa = T(0)) noexcept
      : policy_(std::move(policy)), alpha_(alpha), beta_(beta), kappa_(kappa) {
    if (T(N) + kappa_ <= T(0)) {
      kappa_ = T(3) - T(N);
    }
    lambda_ = alpha_ * alpha_ * (T(N) + kappa_) - T(N);
    computeWeights();
    x_.setZero();
    P_.setIdentity();
    qc_ = T(1);
    R_.setIdentity();
  }

  /// Time-update (predict) step.
  ///
  /// Propagates the state and covariance forward by `dt` seconds using the
  /// discrete-time process model selected by `Policy::ModelTag`.
  /// If `dt ≤ 0` the call is a no-op.
  void predict(T dt) {
    if (dt <= T(0)) return;
    KALMAN_PROFILE_ZONE_NAMED_C("UKF::predict", kalman::profiling::ColorCore);

    using ModelTag = typename Policy::ModelTag;
    const auto F = ProcessModel<T, N, ModelTag>::F(dt);
    const auto Q = ProcessModel<T, N, ModelTag>::Q(dt, qc_);
    x_ = F * x_;
    P_ = F * P_ * F.transpose() + Q;
    P_ = T(0.5) * (P_ + P_.transpose());

#ifndef NDEBUG
    assert(x_.allFinite());
    assert(P_.allFinite());
#endif
  }

  /// Measurement-update (correct) step.
  ///
  /// For linear measurement models (`isMeasurementLinear == true`) the
  /// closed-form KF update is used.  For non-linear models the full UKF
  /// sigma-point update is used.
  void update(const VectorM& z) {
    KALMAN_PROFILE_ZONE_NAMED_C("UKF::update", kalman::profiling::ColorCore);
    MatrixM S;
    MatrixNM K;

    if constexpr (Policy::isMeasurementLinear) {
      const auto Hmat = policy_.H(x_);

      const VectorM z_pred = Hmat * x_;
      VectorM nu = z - z_pred;
      policy_.wrapInnovation(nu);

      S = Hmat * P_ * Hmat.transpose() + R_;

      const MatrixNM C = P_ * Hmat.transpose();
      K = S.ldlt().solve(C.transpose()).transpose();

      x_ = x_ + K * nu;
      P_ = P_ - K * S * K.transpose();
    } else {
      constexpr int ns = 2 * N + 1;

      MatrixNX Xs;
      generateSigmaPoints(x_, P_, Xs);

      MatrixMX Z;
      for (int i = 0; i < ns; ++i) {
        Z.col(i) = policy_.h(Xs.col(i));
      }

      VectorM z_pred = w_m0_ * Z.col(0);
      for (int i = 1; i < ns; ++i) {
        z_pred += w_i_ * Z.col(i);
      }

      VectorM nu = z - z_pred;
      policy_.wrapInnovation(nu);

      S = w_c0_ * (Z.col(0) - z_pred) * (Z.col(0) - z_pred).transpose();
      for (int i = 1; i < ns; ++i) {
        auto diff = Z.col(i) - z_pred;
        S += w_i_ * diff * diff.transpose();
      }
      S += R_;

      MatrixNM C = w_c0_ * (Xs.col(0) - x_) * (Z.col(0) - z_pred).transpose();
      for (int i = 1; i < ns; ++i) {
        auto dx = Xs.col(i) - x_;
        auto dz = Z.col(i) - z_pred;
        C += w_i_ * dx * dz.transpose();
      }

      K = S.ldlt().solve(C.transpose()).transpose();

      x_ = x_ + K * nu;
      P_ = P_ - K * S * K.transpose();
    }

    P_ = T(0.5) * (P_ + P_.transpose());

#ifndef NDEBUG
    assert(x_.allFinite());
    assert(P_.allFinite());
    if (!P_.allFinite()) {
      std::fprintf(stderr, "P non-finite at update (N=%d).\n", N);
      std::fprintf(stderr, "P.diag =");
      for (int i = 0; i < N; ++i) {
        std::fprintf(stderr, " %.6e", P_(i, i));
      }
      std::fprintf(stderr, "\n");
      std::fprintf(stderr, "S.diag = (%.6e,%.6e)\n", S(0, 0), S(1, 1));
      std::fprintf(stderr, "K.norm = %.6e\n", K.norm());
    }
#endif
  }

  /// Convenience: predict then update.
  void filter(T dt, const VectorM& z) {
    KALMAN_PROFILE_ZONE_NAMED_C("UKF::filter", kalman::profiling::ColorCore);
    predict(dt);
    update(z);
  }

  [[nodiscard]] const VectorN& getState() const noexcept { return x_; }
  [[nodiscard]] const MatrixN& getCovariance() const noexcept { return P_; }
  [[nodiscard]] Policy& getPolicy() noexcept { return policy_; }
  [[nodiscard]] const Policy& getPolicy() const noexcept { return policy_; }

  void setState(const VectorN& x) noexcept { x_ = x; }
  void setCovariance(const MatrixN& P) noexcept { P_ = P; }
  void setProcessNoiseSpectralDensity(T qc) noexcept { qc_ = qc; }
  void setMeasurementNoiseCovariance(const MatrixM& R) noexcept { R_ = R; }

 private:
  void generateSigmaPoints(const VectorN& x, const MatrixN& P, MatrixNX& Xs) const {
    KALMAN_PROFILE_ZONE_NAMED_C("UKF::generateSigmaPoints", kalman::profiling::ColorCore);
    constexpr T kReg = T(1e-6);
    const T reg = kReg * std::max(T(1), P.trace() / T(N));
    const MatrixN scaledP = (T(N) + lambda_) * (P + reg * MatrixN::Identity());
    const MatrixN L = scaledP.llt().matrixL();

    Xs.col(0) = x;
    for (int i = 0; i < N; ++i) {
      Xs.col(i + 1) = x + L.col(i);
      Xs.col(i + N + 1) = x - L.col(i);
    }
  }

  void computeWeights() noexcept {
    const T denom = T(N) + lambda_;
    w_m0_ = lambda_ / denom;
    w_c0_ = w_m0_ + (T(1) - alpha_ * alpha_ + beta_);
    w_i_ = T(1) / (T(2) * denom);
  }

  Policy policy_;
  T alpha_, beta_, kappa_, lambda_;
  VectorN x_{};
  MatrixN P_{};
  T qc_;
  MatrixM R_{};
  T w_m0_, w_c0_, w_i_;
};

}  // namespace kalman
