/// @file
/// @brief GlobalMotionEstimator test: recovers a known background translation
///        from two synthetic frames while a foreground blob (inside the
///        excluded ROI) moves differently.

#include "global_motion.hpp"

#include <gtest/gtest.h>

#include <cstdlib>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <random>
#include <vector>

namespace {

cv::Mat makeBackground(int w, int h, unsigned seed) {
  cv::Mat img(h, w, CV_8UC3, cv::Scalar(60, 60, 60));
  std::mt19937 rng(seed);
  std::uniform_int_distribution<int> xu(0, w - 1);
  std::uniform_int_distribution<int> yu(0, h - 1);
  std::uniform_int_distribution<int> ru(2, 5);
  std::uniform_int_distribution<int> lu(100, 220);
  for (int i = 0; i < 250; ++i) {
    const cv::Point c(xu(rng), yu(rng));
    const int r = ru(rng);
    cv::circle(img, c, r, cv::Scalar(lu(rng), lu(rng), lu(rng)), -1, cv::LINE_AA);
  }
  return img;
}

void drawForeground(cv::Mat& img, const cv::Rect& roi, int val) {
  cv::rectangle(img, roi, cv::Scalar(val, val, 255), cv::FILLED);
}

void shiftBy(cv::Mat& img, const cv::Point2f& d) {
  cv::Mat_<double> M = (cv::Mat_<double>(2, 3) << 1.0, 0.0, d.x, 0.0, 1.0, d.y);
  cv::warpAffine(img, img, M, img.size(), cv::INTER_LINEAR, cv::BORDER_REPLICATE);
}

}  // namespace

TEST(GlobalMotionEstimator, RecoversBackgroundTranslation) {
  constexpr int W = 640;
  constexpr int H = 480;
  const cv::Point2f shift(12.0f, -8.0f);
  const cv::Rect roi(200, 150, 80, 80);

  auto frame0 = makeBackground(W, H, 11);
  auto frame1 = makeBackground(W, H, 11);
  shiftBy(frame1, shift);
  drawForeground(frame0, roi, 0);
  drawForeground(frame1, cv::Rect(roi.x + 60, roi.y + 40, roi.width, roi.height), 0);

  GlobalMotionEstimator::Config cfg;
  cfg.downsample = 1.0;
  cfg.ransac_threshold = 1.0;
  GlobalMotionEstimator est(cfg);

  auto first = est.track(frame0, cv::Rect());
  EXPECT_FALSE(first.valid) << "First frame has no previous frame; no motion expected";

  auto second = est.track(frame1, roi);
  ASSERT_TRUE(second.valid) << "Background translation should be estimated";
  EXPECT_NEAR(second.displacement.x, shift.x, 1.0) << "x displacement mismatch";
  EXPECT_NEAR(second.displacement.y, shift.y, 1.0) << "y displacement mismatch";
  EXPECT_GE(second.inliers, 8);
}

TEST(GlobalMotionEstimator, ResetClearsState) {
  constexpr int W = 640;
  constexpr int H = 480;
  const cv::Point2f shift(5.0f, 5.0f);

  auto frame0 = makeBackground(W, H, 22);
  auto frame1 = makeBackground(W, H, 22);
  shiftBy(frame1, shift);

  GlobalMotionEstimator::Config cfg;
  cfg.downsample = 1.0;
  cfg.ransac_threshold = 1.0;
  GlobalMotionEstimator est(cfg);

  est.track(frame0, cv::Rect());
  auto first = est.track(frame1, cv::Rect());
  ASSERT_TRUE(first.valid);

  est.reset();
  auto after_reset = est.track(frame0, cv::Rect());
  EXPECT_FALSE(after_reset.valid) << "Reset should force the estimator to re-seed corners";
}