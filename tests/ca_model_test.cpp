#include "kalman/detail/ca_model.hpp"

#include <gtest/gtest.h>

#include <Eigen/Eigenvalues>

namespace {

using kalman::detail::caNoise;
using kalman::detail::caNoiseAxis;
using kalman::detail::caTransition;
using kalman::detail::caTransitionAxis;

using Matrix6 = Eigen::Matrix<float, 6, 6>;

}  // namespace

TEST(CaModel, TransitionIsIdentityAtZeroDt) {
  EXPECT_TRUE(caTransition(0.0f).isApprox(Matrix6::Identity(), 1e-6f));
  EXPECT_TRUE(
      caTransitionAxis(0.0f).isApprox(Eigen::Matrix3f::Identity(), 1e-6f));
}

TEST(CaModel, TransitionScattersPerAxisBlocks) {
  const float dt = 0.025f;
  const Matrix6 F = caTransition(dt);
  // u-axis indices (0, 2, 4): u' = u + vu·dt + au·dt²/2, vu' = vu + au·dt.
  EXPECT_FLOAT_EQ(F(0, 0), 1.0f);
  EXPECT_FLOAT_EQ(F(0, 2), dt);
  EXPECT_FLOAT_EQ(F(0, 4), 0.5f * dt * dt);
  EXPECT_FLOAT_EQ(F(2, 2), 1.0f);
  EXPECT_FLOAT_EQ(F(2, 4), dt);
  EXPECT_FLOAT_EQ(F(4, 4), 1.0f);
  // v-axis indices (1, 3, 5).
  EXPECT_FLOAT_EQ(F(1, 3), dt);
  EXPECT_FLOAT_EQ(F(1, 5), 0.5f * dt * dt);
  EXPECT_FLOAT_EQ(F(3, 5), dt);
  // Axes are independent: no cross terms between {0,2,4} and {1,3,5}.
  for (int i : {0, 2, 4}) {
    for (int j : {1, 3, 5}) {
      EXPECT_FLOAT_EQ(F(i, j), 0.0f);
      EXPECT_FLOAT_EQ(F(j, i), 0.0f);
    }
  }
}

TEST(CaModel, TransitionIsASemigroup) {
  // Exact for piecewise constant acceleration: F(a)·F(b) = F(a+b).
  const float a = 0.025f;
  const float b = 0.1f;
  const Matrix6 Fab = caTransition(a) * caTransition(b);
  EXPECT_TRUE(Fab.isApprox(caTransition(a + b), 1e-5f));
}

TEST(CaModel, NoiseIsSymmetricPositiveSemidefinite) {
  for (const float dt : {1e-5f, 1e-3f, 0.025f, 0.2f, 1.0f}) {
    const Matrix6 Q = caNoise(dt, 2.0e5f);
    EXPECT_TRUE(Q.isApprox(Q.transpose(), 1e-5f));
    const Eigen::SelfAdjointEigenSolver<Matrix6> es(Q);
    const float lam_min = es.eigenvalues().minCoeff();
    const float lam_max = es.eigenvalues().maxCoeff();
    EXPECT_GE(lam_min, -1e-4f * std::max(1.0f, lam_max));
  }
}

TEST(CaModel, NoiseVanishesAtZeroDt) {
  EXPECT_EQ(caNoise(0.0f, 2.0e5f), Matrix6::Zero());
}

TEST(CaModel, NoiseMatchesGramianQuadrature) {
  // Q3(dt) = q ∫₀^dt g(τ)g(τ)ᵀ dτ with g(τ) = F3(τ)·e₃ = [τ²/2, τ, 1]ᵀ.
  const float dt = 0.025f;
  const float q = 2.0e5f;
  const Eigen::Matrix3f analytic = caNoiseAxis(dt, q);

  constexpr int kSteps = 20000;
  const float h = dt / static_cast<float>(kSteps);
  Eigen::Matrix3f numeric = Eigen::Matrix3f::Zero();
  for (int k = 0; k < kSteps; ++k) {
    const float t0 = (static_cast<float>(k) + 0.5f) * h;
    const Eigen::Vector3f g = caTransitionAxis(t0).col(2);
    numeric += (g * g.transpose()) * (q * h);
  }
  EXPECT_TRUE(numeric.isApprox(analytic, 1e-3f));
}

TEST(CaModel, NoiseScattersPerAxisBlocks) {
  const float dt = 0.025f;
  const float q = 2.0e5f;
  const Matrix6 Q = caNoise(dt, q);
  const Eigen::Matrix3f Q3 = caNoiseAxis(dt, q);
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) {
      EXPECT_FLOAT_EQ(Q(2 * i, 2 * j), Q3(i, j));
      EXPECT_FLOAT_EQ(Q(2 * i + 1, 2 * j + 1), Q3(i, j));
      EXPECT_FLOAT_EQ(Q(2 * i, 2 * j + 1), 0.0f);
    }
  }
}
