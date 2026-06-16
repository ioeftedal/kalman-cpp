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
#include <numbers>

// Provides the default PanTiltPolicy and the ProcessModel specialisations
// for N=9 (CA) and N=6 (CV) with the default tag.
#include "kalman/pan_tilt_measurement_model.hpp"

namespace kalman {

/// Unscented Kalman Filter for state dimension N and measurement dimension M.
///
/// The UKF propagates a set of deterministically chosen sigma points through
/// the non-linear process and measurement functions, recovering the posterior
/// mean and covariance to second-order accuracy.
///
/// @tparam T    Floating-point scalar type (e.g. `float`, `double`).
/// @tparam N    State vector dimension.
/// @tparam M    Measurement vector dimension.
/// @tparam Policy  Policy class providing the measurement model `h(x)` and
///             `wrapInnovation(nu)`, plus the `ModelTag` used to select the
///             process model at compile time.  Default: `PanTiltPolicy`.
///
/// State vector (N=9, pan/tilt model):
///   x = [x, y, z, vx, vy, vz, ax, ay, az]ᵀ
///
/// Coordinate convention (camera-centric):
///   +x : camera right
///   +y : camera up
///   +z : camera forward (optical axis)
///
/// Process model (default):  constant acceleration driven by Wiener-process
/// jerk with power spectral density q_c (continuous-time).  The discrete-time
/// matrices F(dt) and Q(dt, q_c) are selected at compile time via
/// `Policy::ModelTag` and `ProcessModel<T, N, ModelTag>`.
///
/// Measurement model:  non-linear bearing-only pan/tilt (default), or linear
/// 2-D pixel extraction — determined by the Policy parameter.
template <typename T, int N, int M, typename Policy = PanTiltPolicy>
class UnscentedKalmanFilter {
  static_assert(N > 0, "State dimension N must be positive");
  static_assert(M > 0, "Measurement dimension M must be positive");

 public:
  /// N-dimensional column vector (state).
  using VectorN = Eigen::Matrix<T, N, 1>;
  /// N×N matrix (state covariance, process noise).
  using MatrixN = Eigen::Matrix<T, N, N>;
  /// M-dimensional column vector (measurement).
  using VectorM = Eigen::Matrix<T, M, 1>;
  /// M×M matrix (measurement noise covariance).
  using MatrixM = Eigen::Matrix<T, M, M>;
  /// N×M cross-covariance matrix.
  using MatrixNM = Eigen::Matrix<T, N, M>;
  /// N × (2N+1) sigma-point matrix (columns are sigma points).
  using MatrixNX = Eigen::Matrix<T, N, 2 * N + 1>;
  /// M × (2N+1) predicted-measurement sigma-point matrix.
  using MatrixMX = Eigen::Matrix<T, M, 2 * N + 1>;
  /// (2N+1)-dimensional weight vector.
  using VectorW = Eigen::Matrix<T, 2 * N + 1, 1>;

  /// Construct an UnscentedKalmanFilter.
  ///
  /// @param alpha  Primary scaling parameter — controls sigma-point spread.
  ///               Typical range (0, 1].  Smaller values reduce higher-order
  ///               non-linear effects.
  /// @param beta   Prior knowledge parameter.  β = 2 is optimal for Gaussian
  ///               distributions.
  /// @param kappa  Secondary scaling parameter.  Often set to 0 or 3 − N.
  explicit UnscentedKalmanFilter(T alpha = T(1e-3), T beta = T(2), T kappa = T(0)) noexcept
      : alpha_(alpha), beta_(beta), kappa_(kappa) {
    // Ensure positive-definiteness of (N+lambda)*P for the Cholesky
    // in generateSigmaPoints.  The Van der Merwe default is kappa = 3 - N.
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

  /// Perform the time-update (predict) step.
  ///
  /// Propagates the state and covariance forward by `dt` seconds using the
  /// discrete-time process model:
  ///   x = F(dt) · x
  ///   P = F(dt) · P · F(dt)ᵀ + Q(dt, q_c)
  ///
  /// The process model matrices F and Q are dispatched at compile time based
  /// on `Policy::ModelTag` and the state dimension N.
  ///
  /// @param dt  Time step (seconds).  If `dt ≤ 0` the call is a no-op.
  void predict(T dt) {
    if (dt <= T(0)) return;

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

  /// Perform the measurement-update (correct) step.
  ///
  /// For linear measurement models (`Policy::isMeasurementLinear == true`)
  /// the standard closed-form Kalman-filter update is used (exact, cheaper).
  /// For non-linear models the full UKF update is used:
  ///   1. Generate 2N+1 sigma points from the current state x and covariance P.
  ///   2. Propagate each sigma point through the measurement function h(·).
  ///   3. Compute the predicted measurement mean z_pred.
  ///   4. Compute the innovation covariance S and cross-covariance C.
  ///   5. Compute the Kalman gain K = C · S⁻¹.
  ///   6. Update: x += K · (z − z_pred),  P −= K · S · Kᵀ.
  ///
  /// Innovation is optionally wrapped (e.g. for angular residuals) via
  /// `Policy::wrapInnovation()`.
  ///
  /// @param z  Measurement vector of dimension M.
  void update(const VectorM& z) {
    MatrixM S;
    MatrixNM K;

    if constexpr (Policy::isMeasurementLinear) {
      // Linear measurement model — standard KF update (closed-form, exact)
      const auto Hmat = Policy::H(x_);

      const VectorM z_pred = Hmat * x_;
      VectorM nu = z - z_pred;
      Policy::wrapInnovation(nu);

      S = Hmat * P_ * Hmat.transpose() + R_;

      const MatrixNM C = P_ * Hmat.transpose();
      K = S.ldlt().solve(C.transpose()).transpose();

      x_ = x_ + K * nu;
      P_ = P_ - K * S * K.transpose();
    } else {
      // Non-linear measurement model — UKF sigma-point update
      constexpr int ns = 2 * N + 1;

      MatrixNX Xs;
      generateSigmaPoints(x_, P_, Xs);

      MatrixMX Z;
      for (int i = 0; i < ns; ++i) {
        Z.col(i) = Policy::h(Xs.col(i));
      }

      VectorM z_pred = w_m0_ * Z.col(0);
      for (int i = 1; i < ns; ++i) {
        z_pred += w_i_ * Z.col(i);
      }

      VectorM nu = z - z_pred;
      Policy::wrapInnovation(nu);

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
  /// @param dt  Time step (seconds).
  /// @param z   Measurement vector.
  void filter(T dt, const VectorM& z) {
    predict(dt);
    update(z);
  }

  /// @return  Read-only reference to the current state estimate.
  [[nodiscard]] const VectorN& getState() const noexcept { return x_; }
  /// @return  Read-only reference to the current state covariance.
  [[nodiscard]] const MatrixN& getCovariance() const noexcept { return P_; }

  /// Overwrite the state estimate.
  void setState(const VectorN& x) noexcept { x_ = x; }
  /// Overwrite the state covariance.
  void setCovariance(const MatrixN& P) noexcept { P_ = P; }
  /// Set the continuous-time process-noise spectral density q_c.
  /// @param qc  Power spectral density of the Wiener-process jerk (or velocity).
  void setProcessNoiseSpectralDensity(T qc) noexcept { qc_ = qc; }
  /// Set the measurement noise covariance matrix.
  /// @param R  M×M symmetric positive-definite covariance matrix.
  void setMeasurementNoiseCovariance(const MatrixM& R) noexcept { R_ = R; }

 private:
  /// Generate the 2N+1 sigma-point matrix Xs from the current state and
  /// covariance using the scaled unscented transform.
  ///
  /// Columns of Xs:
  ///   Xs[:, 0]   = x
  ///   Xs[:, i+1] = x + √[(N+λ)·P]_i     (i = 0 … N−1)
  ///   Xs[:, N+1+i] = x − √[(N+λ)·P]_i
  ///
  /// The matrix square root is computed via Cholesky decomposition.  A
  /// relative regularisation (kReg × mean diagonal of P) is added to P to
  /// keep the decomposition well-conditioned regardless of P's condition
  /// number, without over-regularising large P or under-regularising small P.
  ///
  /// @param x   State mean.
  /// @param P   State covariance.
  /// @param Xs  [out] On return, filled with sigma points as columns.
  void generateSigmaPoints(const VectorN& x, const MatrixN& P, MatrixNX& Xs) const {
    // Relative regularisation — scales with P's magnitude so that it neither
    // over-regularises (large P early on) nor under-regularises (small P at
    // steady state).
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

  /// Compute the sigma-point weights from the current scaling parameters.
  ///
  /// Mean weight (sigma-point 0):  w_m⁰ = λ / (N + λ)
  /// Covariance weight (sigma-point 0):  w_c⁰ = w_m⁰ + (1 − α² + β)
  /// Remaining weights:  w_i = 1 / [2 (N + λ)]
  void computeWeights() noexcept {
    const T denom = T(N) + lambda_;
    w_m0_ = lambda_ / denom;
    w_c0_ = w_m0_ + (T(1) - alpha_ * alpha_ + beta_);
    w_i_ = T(1) / (T(2) * denom);
  }

  T alpha_, beta_, kappa_, lambda_;
  VectorN x_{};
  MatrixN P_{};
  T qc_;
  MatrixM R_{};
  T w_m0_, w_c0_, w_i_;
};

}  // namespace kalman
