/// @file
/// @brief Fixed-size linear Kalman filter core with direct state observation.
///
/// The measurement model is `z = x[0..M) + noise`, i.e. `H = [I_M 0]`: the
/// detector reports the first M state components (here: position) directly.
/// That structural assumption is exploited instead of being re-derived from a
/// general `H` every frame — `S = P₀₀ + R`, `K = P(:,0:M)·S⁻¹` — so no `H`
/// is ever materialised and the update is a fixed-size LDLT of an M×M matrix.
///
/// Everything is linear, so there are no sigma points, no Jacobians and no
/// policy dispatch — just matrix algebra on compile-time-sized Eigen types
/// (which vectorise to NEON/SSE with no heap traffic).  The covariance update
/// is the Joseph form `P = (I−KH)P(I−KH)ᵀ + KRKᵀ`, which preserves symmetry
/// and positive-definiteness to working precision regardless of round-off.
///
/// Numerical contract: a decomposition failure, a non-finite NIS, or a
/// negative NIS (which can only mean `S` lost positive-definiteness) all
/// report as a rejected update with the state left untouched — corrupt input
/// degrades to "no information applied", never to a corrupted state.

#pragma once

#include <Eigen/Cholesky>
#include <Eigen/Core>

#include <cmath>
#include <limits>

namespace kalman::detail {

/// Linear KF over an N-dimensional state whose first M components are
/// measured directly, with innovation gating.
template <int N, int M>
class LinearFilter {
  static_assert(M <= N, "measurement dimension cannot exceed state dimension");

 public:
  using VectorN = Eigen::Matrix<float, N, 1>;
  using MatrixN = Eigen::Matrix<float, N, N>;
  using VectorM = Eigen::Matrix<float, M, 1>;
  using MatrixM = Eigen::Matrix<float, M, M>;
  using MatrixNM = Eigen::Matrix<float, N, M>;

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

  /// Measurement update for `z = x[0..M) + w`, `Var(w) = r·I`.
  /// Returns false (leaving the state untouched) when the normalised
  /// innovation squared exceeds `gate_chi2`; the rejected NIS is still
  /// recorded for diagnostics.
  bool updatePosition(const VectorM& z, float r, float gate_chi2) {
    const VectorM nu = z - x_.template head<M>();

    // S = H P Hᵀ + R with H = [I 0]  ⇒  S = P₀₀ + r·I.
    MatrixM S = P_.template topLeftCorner<M, M>();
    S.diagonal().array() += r;

    const Eigen::LDLT<MatrixM> ldlt(S);
    if (ldlt.info() != Eigen::Success) {
      last_nis_ = std::numeric_limits<float>::infinity();
      return false;
    }
    last_nis_ = nu.dot(ldlt.solve(nu));
    // A non-finite NIS means the measurement or the state is unusable (NaN
    // detection, corrupt covariance); report it as +inf so no NaN escapes the
    // library.  The `!(...)` form below rejects NaN as well, and a negative
    // NIS means S is not positive-definite, so the solve cannot be trusted.
    if (!std::isfinite(last_nis_)) {
      last_nis_ = std::numeric_limits<float>::infinity();
    }
    if (!(last_nis_ >= 0.0f) || !(last_nis_ < gate_chi2)) {
      return false;
    }

    // K = P Hᵀ S⁻¹ = P(:,0:M)·S⁻¹.  Concrete result type: an `auto` here
    // would hold an expression referencing the LDLT and the temporary
    // identity below, reading freed memory once either dies.
    const MatrixM s_inv = ldlt.solve(MatrixM::Identity());
    const MatrixNM K = P_.template leftCols<M>() * s_inv;

    x_ += K * nu;

    // I − KH = I − [K 0]: subtract K from the leading M columns only.
    MatrixN I_KH = MatrixN::Identity();
    I_KH.template leftCols<M>() -= K;
    P_ = I_KH * P_ * I_KH.transpose() + r * (K * K.transpose());
    P_ = 0.5f * (P_ + P_.transpose());
    return true;
  }

  [[nodiscard]] const VectorN& state() const noexcept { return x_; }
  [[nodiscard]] const MatrixN& covariance() const noexcept { return P_; }
  /// NIS of the most recent update attempt, accepted or rejected.
  [[nodiscard]] float lastNis() const noexcept { return last_nis_; }

  /// Replaces the state/covariance; the previous NIS described the old track
  /// and is cleared with it.
  void setState(const VectorN& x) noexcept {
    x_ = x;
    last_nis_ = 0.0f;
  }
  void setCovariance(const MatrixN& P) noexcept {
    P_ = P;
    last_nis_ = 0.0f;
  }

 private:
  VectorN x_{};
  MatrixN P_{};
  float last_nis_ = 0.0f;
};

}  // namespace kalman::detail
