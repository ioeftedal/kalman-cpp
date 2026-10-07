/// @file
/// @brief Fixed-size linear Kalman filter core: predict, gated update.
///
/// Everything here is linear, so there are no sigma points, no Jacobians and
/// no policy dispatch — just matrix algebra on compile-time-sized Eigen types
/// (which vectorise to NEON/SSE with no heap traffic).  The covariance update
/// is the Joseph form `P = (I−KH)P(I−KH)ᵀ + KRKᵀ`, which preserves symmetry
/// and positive-definiteness to working precision regardless of round-off.

#pragma once

#include <Eigen/Cholesky>
#include <Eigen/Core>

namespace kalman::detail {

/// Linear KF over an N-dimensional state with an M-dimensional measurement.
template <int N, int M>
class LinearFilter {
 public:
  using VectorN = Eigen::Matrix<float, N, 1>;
  using MatrixN = Eigen::Matrix<float, N, N>;
  using VectorM = Eigen::Matrix<float, M, 1>;
  using MatrixM = Eigen::Matrix<float, M, M>;
  using MatrixNM = Eigen::Matrix<float, N, M>;
  using MatrixMN = Eigen::Matrix<float, M, N>;

  LinearFilter() {
    x_.setZero();
    P_.setIdentity();
  }

  /// Time update: `x = Fx`, `P = FPFᵀ + Q`.
  void predict(const MatrixN& F, const MatrixN& Q) {
    x_ = F * x_;
    P_ = F * P_ * F.transpose() + Q;
    P_ = 0.5f * (P_ + P_.transpose());
  }

  /// Measurement update with innovation gating (`z = H·x`, so `H` is M×N).
  /// Returns false (leaving the state untouched) when the normalised
  /// innovation squared exceeds `gate_chi2`; the rejected NIS is still
  /// recorded for diagnostics.
  bool update(const VectorM& z, const MatrixM& R, const MatrixMN& H,
              float gate_chi2) {
    const VectorM nu = z - H * x_;
    const MatrixM S = H * P_ * H.transpose() + R;

    const Eigen::LDLT<MatrixM> ldlt(S);
    last_nis_ = nu.dot(ldlt.solve(nu));
    if (!(last_nis_ < gate_chi2)) {
      return false;
    }

    // K = P Hᵀ S⁻¹, materialised from the solve (an `auto` here would keep a
    // reference to the temporary LDLT and read freed memory).
    const MatrixNM K = ldlt.solve((P_ * H.transpose()).transpose()).transpose();
    x_ = x_ + K * nu;

    const MatrixN I_KH = MatrixN::Identity() - K * H;
    P_ = I_KH * P_ * I_KH.transpose() + K * R * K.transpose();
    P_ = 0.5f * (P_ + P_.transpose());
    return true;
  }

  [[nodiscard]] const VectorN& state() const noexcept { return x_; }
  [[nodiscard]] const MatrixN& covariance() const noexcept { return P_; }
  /// NIS of the most recent update attempt, accepted or rejected.
  [[nodiscard]] float lastNis() const noexcept { return last_nis_; }

  void setState(const VectorN& x) noexcept { x_ = x; }
  void setCovariance(const MatrixN& P) noexcept { P_ = P; }

 private:
  VectorN x_{};
  MatrixN P_{};
  float last_nis_ = 0.0f;
};

}  // namespace kalman::detail
