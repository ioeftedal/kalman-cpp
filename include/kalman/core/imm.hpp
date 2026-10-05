/// @file
/// @brief Interacting Multiple-Model (IMM) filter over pixel-space models.
///
/// Mixes a set of linear-Gaussian motion models (e.g. CV / CA / CT) via mode
/// probabilities and a Markov transition matrix.  All models share the 2-D
/// position measurement and are mixed in the common 4-D subspace [u, v, vu, vv];
/// model-specific extras (CA accelerations, CT turn rate) are re-seeded after
/// each mixing step.

#pragma once

#include <Eigen/Cholesky>
#include <Eigen/Core>
#include <Eigen/Eigenvalues>
#include <algorithm>
#include <array>
#include <cmath>
#include <tuple>
#include <type_traits>
#include <utility>

#include "kalman/core/constants.hpp"
#include "kalman/core/filter.hpp"
#include "kalman/kinematics/ct_pixel.hpp"
#include "kalman/models/pixel.hpp"

namespace kalman {

// ---------------------------------------------------------------------------
//  Per-model descriptor: pixel-policy tag + UKF type + state dimension.
// ---------------------------------------------------------------------------

template <typename T, typename Tag>
struct ImmPixelModel {
  static constexpr int dim = PixelStateDim<Tag>::value;
  using TagType = Tag;
  using Filter = UnscentedKalmanFilter<T, dim, 2, PixelPolicyBase<Tag>>;
};

// ---------------------------------------------------------------------------
//  IMM filter
// ---------------------------------------------------------------------------

template <typename T, typename... Models>
class ImmFilter {
 public:
  using Scalar = T;
  static constexpr int NumModels = static_cast<int>(sizeof...(Models));
  static constexpr int MeasDim = 2;
  static constexpr int CommonDim = 4;

  using VectorC = Eigen::Matrix<T, CommonDim, 1>;
  using MatrixC = Eigen::Matrix<T, CommonDim, CommonDim>;
  using VectorM = Eigen::Matrix<T, MeasDim, 1>;
  using MatrixM = Eigen::Matrix<T, MeasDim, MeasDim>;
  using ModeVec = Eigen::Matrix<T, NumModels, 1>;
  using ModeMatrix = Eigen::Matrix<T, NumModels, NumModels>;

  ImmFilter() {
    mu_.setConstant(T(1) / T(NumModels));
    cbar_ = mu_;
    fused_.setZero();
    fused_cov_.setIdentity();
    setTransitionProbabilities(T(0.02));
  }

  // ---------------------------------------------------------------------
  //  Configuration
  // ---------------------------------------------------------------------

  /// Largest `window` accepted by `setAdaptiveNoise`. The adaptive-noise
  /// history is a fixed-capacity ring buffer rather than a std::deque so that
  /// update() never allocates; longer windows are clamped to this value.
  static constexpr int kMaxAdaptWindow = 256;

  /// Uniform off-diagonal transition probability `p`; diagonal = 1 - p·(N-1).
  void setTransitionProbabilities(T p) {
    Pi_.setConstant(p);
    for (int j = 0; j < NumModels; ++j) {
      Pi_(j, j) = T(1) - p * T(NumModels - 1);
    }
  }

  void setModeProbabilities(const ModeVec& mu) { mu_ = mu; }

  void setProcessNoiseSpectralDensities(std::initializer_list<T> qc) {
    int i = 0;
    for (const T q : qc) {
      qc_[i] = q;
      setQcAt(i, q);
      ++i;
    }
  }

  /// Online, innovation-driven measurement-noise adaption (magic-number-free
  /// tuning for the fixed model set).  `target` is the expected normalised
  /// innovation (== measurement dimension).
  void setAdaptiveNoise(bool on, int window = 30, T target = T(2)) {
    adaptive_ = on;
    adapt_window_ = std::min(std::max(window, 1), kMaxAdaptWindow);
    adapt_target_ = target;
    gate_head_ = 0;
    gate_count_ = 0;
  }

  /// Turn-rate prior (rad/s) injected into the CT model before every predict.
  /// Turn rate is unobservable from the (linear) position update, so it must
  /// be supplied externally -- e.g. from the rotation of the optical-flow
  /// velocity estimate in the tracking app.
  void setTurnRatePrior(T w_rad_s) { turn_prior_ = std::min(std::max(w_rad_s, T(-4)), T(4)); }

  // ---------------------------------------------------------------------
  //  Lifecycle
  // ---------------------------------------------------------------------

  /// Initialise all models from a common 4-D estimate (extras are zeroed).
  void initialize(const VectorC& x4, const MatrixC& P4, T extra_var = T(1e4)) {
    extra_var_ = extra_var;
    for (std::size_t i = 0; i < NumModels; ++i) {
      initModelAt(static_cast<int>(i), x4, P4);
    }
    mu_.setConstant(T(1) / T(NumModels));
    cbar_ = mu_;
    gate_head_ = 0;
    gate_count_ = 0;
    noise_scale_ = T(1);
    fused_ = x4;
    fused_cov_ = P4;
  }

  /// Mix, then time-propagate every model to time t+dt.
  void predict(T dt) {
    // 1. Propagate mode probabilities through the transition matrix.
    for (int j = 0; j < NumModels; ++j) {
      cbar_(j) = T(0);
      for (int i = 0; i < NumModels; ++i) {
        cbar_(j) += Pi_(i, j) * mu_(i);
      }
    }

    // 2. Mix each model's input in the common 4-D subspace.
    for (int j = 0; j < NumModels; ++j) {
      VectorC xmix4 = VectorC::Zero();
      for (int i = 0; i < NumModels; ++i) {
        const T w = Pi_(i, j) * mu_(i) / cbar_(j);
        xmix4 += w * state4At(i);
      }
      MatrixC Pmix4 = MatrixC::Zero();
      for (int i = 0; i < NumModels; ++i) {
        const T w = Pi_(i, j) * mu_(i) / cbar_(j);
        const VectorC d = state4At(i) - xmix4;
        Pmix4 += w * (cov4At(i) + d * d.transpose());
      }
      setStateAt(j, xmix4, Pmix4);
    }

    // 3. Time-propagate each model (CT uses its state-dependent transition).
    for (std::size_t j = 0; j < NumModels; ++j) {
      predictModelAt(static_cast<int>(j), dt);
    }

    // 4. Fused predicted moments for output / search-ROI use.
    recomputeFused(cbar_);
  }

  /// Position measurement update with per-model likelihoods.  Returns true
  /// when the measurement passes the mixture gate, false otherwise (the
  /// caller treats that as occlusion / outlier).  On false the internal state
  /// and mode probabilities are left untouched.
  bool update(const VectorM& z, const MatrixM& R_meas, T gate_threshold) {
    const MatrixM R_eff = adaptive_ ? MatrixM(R_meas * noise_scale_) : R_meas;

    std::array<VectorM, NumModels> nu;
    std::array<MatrixM, NumModels> S;
    std::array<T, NumModels> L;
    VectorM nu_bar = VectorM::Zero();
    MatrixM S_mix = MatrixM::Zero();

    for (int j = 0; j < NumModels; ++j) {
      const VectorC x4 = state4At(j);
      const MatrixC P4 = cov4At(j);
      nu[j] = z - x4.template head<MeasDim>();
      S[j] = P4.template topLeftCorner<MeasDim, MeasDim>() + R_eff;
      L[j] = gaussian(nu[j], S[j]);
      nu_bar += cbar_(j) * nu[j];
    }
    S_mix = -nu_bar * nu_bar.transpose();
    for (int j = 0; j < NumModels; ++j) {
      S_mix += cbar_(j) * (S[j] + nu[j] * nu[j].transpose());
    }

    last_gate_ = innovationDistance(nu_bar, S_mix);
    if (!(last_gate_ < gate_threshold)) {
      return false;
    }

    // 5. Update mode probabilities from the measurement likelihoods.
    T total = T(0);
    for (int j = 0; j < NumModels; ++j) {
      total += cbar_(j) * L[j];
    }
    ModeVec mu_new;
    for (int j = 0; j < NumModels; ++j) {
      mu_new(j) = cbar_(j) * L[j] / total;
    }

    // 6. Adapt measurement noise from the innovation history.
    if (adaptive_) {
      gate_hist_[gate_head_] = last_gate_;
      gate_head_ = (gate_head_ + 1) % kMaxAdaptWindow;
      if (gate_count_ < kMaxAdaptWindow) {
        ++gate_count_;
      }
      updateAdaptiveScale();
    }

    // 7. Per-model position update with the effective noise.
    for (int j = 0; j < NumModels; ++j) {
      updatePositionAt(j, z, R_eff);
    }
    recomputeFused(mu_new);
    mu_ = mu_new;
    return true;
  }

  /// Sequential velocity (optical-flow) update applied to every model.
  void updateVelocity(const VectorM& z_vel, const MatrixM& R_vel) {
    for (int j = 0; j < NumModels; ++j) {
      updateVelocityAt(j, z_vel, R_vel);
    }
    recomputeFused(mu_);
  }

  /// Hard position association from an authoritative localiser (e.g. a deep
  /// single-object tracker): set every model's position to `z` with the given
  /// covariance, bypassing the (gain-limited) Kalman update. The IMM is left
  /// uncorrected between associations, so its kinematic models only take over
  /// (coast) while the localiser reports no valid fix.
  void setPosition(const VectorM& z, const MatrixM& R_meas) {
    for (int j = 0; j < NumModels; ++j) {
      visitModel(j, [&](auto Ic) {
        constexpr std::size_t I = decltype(Ic)::value;
        auto& f = std::get<I>(filters_);
        auto x = f.getState();
        auto P = f.getCovariance();
        x.template head<MeasDim>() = z;
        P.template topLeftCorner<MeasDim, MeasDim>() = R_meas;
        f.setState(x);
        f.setCovariance(P);
      });
    }
    recomputeFused(mu_);
  }

  // ---------------------------------------------------------------------
  //  Accessors
  // ---------------------------------------------------------------------

  [[nodiscard]] const VectorC& getState() const noexcept { return fused_; }
  [[nodiscard]] const MatrixC& getCovariance() const noexcept { return fused_cov_; }
  [[nodiscard]] const ModeVec& modeProbabilities() const noexcept { return mu_; }
  [[nodiscard]] const ModeVec& predictedModeProbabilities() const noexcept { return cbar_; }
  [[nodiscard]] T noiseScale() const noexcept { return noise_scale_; }
  [[nodiscard]] T lastGateDistance() const noexcept { return last_gate_; }
  [[nodiscard]] static constexpr int numModels() noexcept { return NumModels; }

  /// Mix a vector across models by their current mode probabilities.
  template <int R>
  Eigen::Matrix<T, R, 1> mixScalar(const std::array<Eigen::Matrix<T, R, 1>, NumModels>& values) {
    Eigen::Matrix<T, R, 1> out = Eigen::Matrix<T, R, 1>::Zero();
    for (int j = 0; j < NumModels; ++j) {
      out += mu_(j) * values[j];
    }
    return out;
  }

  /// Common 4-D part of model `j`'s state (u, v, vu, vv) -- used for
  /// per-model diagnostics and mode-conditional tracking.
  VectorC modelStateCommon(int j) const { return state4At(j); }

 private:
  // ---------------------------------------------------------------------
  //  Indexed access to the heterogeneous model tuple
  // ---------------------------------------------------------------------

  template <std::size_t I>
  using Model = std::tuple_element_t<I, std::tuple<Models...>>;

  using FilterTuple = std::tuple<typename Models::Filter...>;

  template <typename Fn>
  void visitModel(int idx, Fn&& fn) {
    visitModelImpl(idx, std::index_sequence_for<Models...>{}, std::forward<Fn>(fn));
  }

  template <typename Fn>
  void visitModel(int idx, Fn&& fn) const {
    visitModelImpl(idx, std::index_sequence_for<Models...>{}, std::forward<Fn>(fn));
  }

  template <std::size_t... Is, typename Fn>
  void visitModelImpl(int idx, std::index_sequence<Is...>, Fn&& fn) const {
    (visitOne(idx, std::integral_constant<std::size_t, Is>{}, fn), ...);
  }

  template <std::size_t I, typename Fn>
  void visitOne(int idx, std::integral_constant<std::size_t, I>, Fn& fn) const {
    if (idx == static_cast<int>(I)) {
      fn(std::integral_constant<std::size_t, I>{});
    }
  }

  // ---------------------------------------------------------------------
  //  Per-model helpers
  // ---------------------------------------------------------------------

  void initModelAt(int j, const VectorC& x4, const MatrixC& P4) {
    visitModel(j, [&](auto Ic) {
      constexpr std::size_t I = decltype(Ic)::value;
      constexpr int D = Model<I>::dim;
      auto& f = std::get<I>(filters_);
      Eigen::Matrix<T, D, 1> x = Eigen::Matrix<T, D, 1>::Zero();
      x.template head<CommonDim>() = x4;
      Eigen::Matrix<T, D, D> P = Eigen::Matrix<T, D, D>::Zero();
      P.template topLeftCorner<CommonDim, CommonDim>() = P4;
      for (int k = CommonDim; k < D; ++k) {
        P(k, k) = extra_var_;
      }
      f.setState(x);
      f.setCovariance(P);
    });
  }

  void setQcAt(int j, T q) {
    visitModel(j, [&](auto Ic) {
      std::get<decltype(Ic)::value>(filters_).setProcessNoiseSpectralDensity(q);
    });
  }

  void predictModelAt(int j, T dt) {
    visitModel(j, [&](auto Ic) {
      constexpr std::size_t I = decltype(Ic)::value;
      auto& f = std::get<I>(filters_);
      if constexpr (std::is_same<typename Model<I>::TagType, PixelCTTag>::value) {
        auto x = f.getState();
        x(4) = turn_prior_;
        auto P = f.getCovariance();
        P.row(4).setZero();
        P.col(4).setZero();
        P(4, 4) = turn_var_;
        f.setState(x);
        f.setCovariance(P);
        const auto F = kinematics::computeF_pixel_CT(x, dt);
        const auto Q = kinematics::computeQ_pixel_CT(dt, qc_[I]);
        Eigen::Matrix<T, Model<I>::dim, Model<I>::dim> Pf = F * P * F.transpose() + Q;
        Pf = T(0.5) * (Pf + Pf.transpose());
        f.setState(F * x);
        f.setCovariance(Pf);
      } else {
        f.predict(dt);
      }
    });
  }

  VectorC state4At(int j) const {
    VectorC out = VectorC::Zero();
    visitModel(j, [&](auto Ic) {
      constexpr std::size_t I = decltype(Ic)::value;
      out = std::get<I>(filters_).getState().template head<CommonDim>();
    });
    return out;
  }

  MatrixC cov4At(int j) const {
    MatrixC out = MatrixC::Zero();
    visitModel(j, [&](auto Ic) {
      constexpr std::size_t I = decltype(Ic)::value;
      out = std::get<I>(filters_).getCovariance().template topLeftCorner<CommonDim, CommonDim>();
    });
    return out;
  }

  void setStateAt(int j, const VectorC& xmix4, const MatrixC& Pmix4) {
    visitModel(j, [&](auto Ic) {
      constexpr std::size_t I = decltype(Ic)::value;
      constexpr int D = Model<I>::dim;
      auto& f = std::get<I>(filters_);
      Eigen::Matrix<T, D, 1> x = Eigen::Matrix<T, D, 1>::Zero();
      x.template head<CommonDim>() = xmix4;
      Eigen::Matrix<T, D, D> P = Eigen::Matrix<T, D, D>::Zero();
      P.template topLeftCorner<CommonDim, CommonDim>() = Pmix4;
      for (int k = CommonDim; k < D; ++k) {
        P(k, k) = extra_var_;
      }
      f.setState(x);
      f.setCovariance(P);
    });
  }

  void updatePositionAt(int j, const VectorM& z, const MatrixM& R) {
    visitModel(j, [&](auto Ic) {
      constexpr std::size_t I = decltype(Ic)::value;
      auto& f = std::get<I>(filters_);
      Eigen::Matrix<T, MeasDim, Model<I>::dim> H = Eigen::Matrix<T, MeasDim, Model<I>::dim>::Zero();
      H(0, 0) = T(1);
      H(1, 1) = T(1);
      linearUpdate(f, z, H, R);
    });
  }

  void updateVelocityAt(int j, const VectorM& z, const MatrixM& R) {
    visitModel(j, [&](auto Ic) {
      constexpr std::size_t I = decltype(Ic)::value;
      auto& f = std::get<I>(filters_);
      Eigen::Matrix<T, MeasDim, Model<I>::dim> H = Eigen::Matrix<T, MeasDim, Model<I>::dim>::Zero();
      H(0, 2) = T(1);
      H(1, 3) = T(1);
      linearUpdate(f, z, H, R);
    });
  }

  // `N` is the per-model state dimension, deduced from `H` (callers pass a
  // fixed-size MeasDim x N measurement Jacobian). Deducing it keeps the gain
  // stack-allocated -- a Dynamic-typed K would malloc on every update, which
  // breaks the allocation-free guarantee real-time consumers depend on.
  template <typename FilterType, int N>
  static void linearUpdate(FilterType& f, const VectorM& z, const Eigen::Matrix<T, MeasDim, N>& H,
                           const MatrixM& R) {
    auto x = f.getState();
    auto P = f.getCovariance();
    const VectorM nu = z - H * x;
    const MatrixM S = R + H * P * H.transpose();
    // Materialise the LDLT solve: `auto K = S.ldlt().solve(...).transpose()` would
    // keep a reference to a temporary LDLT and read freed memory on K * nu.
    const Eigen::Matrix<T, MeasDim, N> Kt = S.ldlt().solve((P * H.transpose()).transpose());
    const Eigen::Matrix<T, N, MeasDim> K = Kt.transpose();
    x = x + K * nu;
    P = P - K * S * K.transpose();
    enforcePDS(P);
    f.setState(x);
    f.setCovariance(P);
  }

  // ---------------------------------------------------------------------
  //  Likelihoods, gating, fusion
  // ---------------------------------------------------------------------

  static T gaussian(const VectorM& nu, const MatrixM& S) {
    const auto inv_nu = S.ldlt().solve(nu);
    const T d2 = nu.dot(inv_nu);
    const T det = std::max(S.determinant(), T(1e-12));
    const T denom = T(2) * kalman::constants::pi_v<T> * std::sqrt(det);
    return std::exp(-T(0.5) * d2) / denom;
  }

  static T innovationDistance(const VectorM& nu, const MatrixM& S) {
    const auto inv_nu = S.ldlt().solve(nu);
    return nu.dot(inv_nu);
  }

  void recomputeFused(const ModeVec& weights) {
    VectorC xbar = VectorC::Zero();
    for (int j = 0; j < NumModels; ++j) {
      xbar += weights(j) * state4At(j);
    }
    MatrixC Pbar = MatrixC::Zero();
    for (int j = 0; j < NumModels; ++j) {
      const VectorC d = state4At(j) - xbar;
      Pbar += weights(j) * (cov4At(j) + d * d.transpose());
    }
    enforcePDS(Pbar);
    fused_ = xbar;
    fused_cov_ = Pbar;
  }

  /// Project `P` onto the symmetric positive-(semi)definite cone. The IMM's
  /// covariance is the sum of model covariances, which is PSD in exact
  /// arithmetic; under hard position associations (where covariance blocks are
  /// overwritten independently of the model cross-covariances) tiny numerical
  /// indefiniteness can creep in and must not propagate into gating/drawing.
  template <int N>
  static void enforcePDS(Eigen::Matrix<T, N, N>& P) {
    P = T(0.5) * (P + P.transpose());
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix<T, N, N>> es(P);
    const auto lam = es.eigenvalues().cwiseMax(T(1e-6));
    P = es.eigenvectors() * lam.asDiagonal() * es.eigenvectors().transpose();
  }

  void updateAdaptiveScale() {
    if (gate_count_ < adapt_window_) {
      return;
    }
    // Walk the live entries of the ring buffer. Once the buffer has wrapped,
    // the oldest sample is at gate_head_; before that it sits at index 0.
    T mean = T(0);
    const int start = (gate_count_ == kMaxAdaptWindow) ? gate_head_ : 0;
    for (int i = 0; i < gate_count_; ++i) {
      mean += gate_hist_[(start + i) % kMaxAdaptWindow];
    }
    mean /= static_cast<T>(gate_count_);
    const T lo = adapt_target_ * T(0.5);
    const T hi = adapt_target_ * T(1.5);
    if (mean > hi) {
      noise_scale_ *= T(1.15);
    } else if (mean < lo) {
      noise_scale_ /= T(1.15);
    }
    noise_scale_ = std::min(std::max(noise_scale_, T(1e-3)), T(1e3));
  }

  FilterTuple filters_;
  ModeMatrix Pi_;
  ModeVec mu_;
  ModeVec cbar_;
  std::array<T, NumModels> qc_{};
  T extra_var_ = T(1e4);
  T turn_prior_ = T(0);
  T turn_var_ = T(0.01);
  bool adaptive_ = false;
  int adapt_window_ = 30;
  T adapt_target_ = T(2);
  T noise_scale_ = T(1);
  T last_gate_ = T(0);
  // Ring buffer of recent normalised innovation distances. `gate_head_` is the
  // next write index; `gate_count_` is how many entries are live.
  std::array<T, kMaxAdaptWindow> gate_hist_{};
  int gate_head_ = 0;
  int gate_count_ = 0;
  VectorC fused_;
  MatrixC fused_cov_;
};

}  // namespace kalman
