// TemplateMatcher (appearance) tests: reacquires a moving target inside a
// search ROI, rejects a genuinely occluded region, prefers the correct crop
// for scoring, and blends template updates.

#include "appearance.hpp"

#include <gtest/gtest.h>

#include <cmath>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

namespace {

using kalman::tracking::AppearanceConfig;
using kalman::tracking::TemplateMatcher;

AppearanceConfig testConfig() {
  AppearanceConfig cfg;
  cfg.template_size = 32;
  cfg.pad_scale = 0.15;
  cfg.min_ncc = 0.5;
  cfg.min_psr = 1.5;
  cfg.update_rate = 0.1;
  cfg.scale_step = 0.15;
  cfg.scale_steps = 3;
  return cfg;
}

cv::Mat makeBackground(int w, int h, unsigned seed) {
  cv::Mat img(h, w, CV_8UC1);
  cv::RNG rng(seed);
  rng.fill(img, cv::RNG::UNIFORM, 0, 255);
  cv::GaussianBlur(img, img, cv::Size(3, 3), 0.5);
  return img;
}

// Self-similar radial blob with a couple of feature dots. Resampling it at a
// different scale yields nearly the same image, so the NCC template match is
// robust to the target's changing apparent size.
void drawBlob(cv::Mat& img, const cv::Rect& r) {
  const cv::Point c(r.x + r.width / 2, r.y + r.height / 2);
  const double rx = r.width / 2.0, ry = r.height / 2.0;
  for (int y = r.y; y < r.y + r.height; ++y) {
    for (int x = r.x; x < r.x + r.width; ++x) {
      const double dx = (x - c.x) / rx, dy = (y - c.y) / ry;
      const double d = std::sqrt(dx * dx + dy * dy);
      const double v = d < 1.0 ? 255.0 - 210.0 * d : 45.0;
      img.at<uchar>(y, x) = cv::saturate_cast<uchar>(v);
    }
  }
  cv::circle(img, c, std::max(2, int(r.width / 6)), cv::Scalar(0), -1);
  cv::circle(img, cv::Point(c.x + r.width / 4, c.y - r.height / 4), std::max(2, int(r.width / 10)),
             cv::Scalar(255), -1);
}

}  // namespace

TEST(TemplateMatcher, ReacquiresTranslatingTarget) {
  constexpr int W = 320;
  constexpr int H = 240;
  const cv::Rect r0(80, 60, 28, 24);
  const cv::Rect r1(r0.x + 15, r0.y - 10, r0.width, r0.height);

  const cv::Mat bg = makeBackground(W, H, 7);
  cv::Mat f0 = bg.clone();
  drawBlob(f0, r0);
  cv::Mat f1 = bg.clone();
  drawBlob(f1, r1);

  TemplateMatcher matcher(testConfig());
  matcher.reset(f0, r0);
  ASSERT_TRUE(matcher.hasTemplate());

  const auto m = matcher.track(f1, cv::Rect(30, 20, 220, 180));

  ASSERT_TRUE(m.valid) << "Shifted target should be reacquired (ncc=" << m.ncc << ", psr=" << m.psr
                       << ")";
  EXPECT_NEAR(m.position.x, r1.x + r1.width / 2.0, 3.0) << "x centre mismatch";
  EXPECT_NEAR(m.position.y, r1.y + r1.height / 2.0, 3.0) << "y centre mismatch";
  EXPECT_GE(m.ncc, testConfig().min_ncc);
}

TEST(TemplateMatcher, ScaleChangeStillMatches) {
  constexpr int W = 320;
  constexpr int H = 240;
  const cv::Rect r0(70, 60, 28, 24);
  const cv::Rect r1(140, 90, 38, 33);

  const cv::Mat bg = makeBackground(W, H, 9);
  cv::Mat f0 = bg.clone();
  drawBlob(f0, r0);
  cv::Mat f1 = bg.clone();
  drawBlob(f1, r1);

  TemplateMatcher matcher(testConfig());
  matcher.reset(f0, r0);

  const auto m = matcher.track(f1, cv::Rect(100, 40, 220, 200));
  ASSERT_TRUE(m.valid) << "Larger target should be matched through the scale sweep (ncc=" << m.ncc
                       << ", psr=" << m.psr << ")";
  EXPECT_NEAR(m.position.x, r1.x + r1.width / 2.0, 4.0) << "x centre mismatch at new scale";
  EXPECT_NEAR(m.position.y, r1.y + r1.height / 2.0, 4.0) << "y centre mismatch at new scale";
}

TEST(TemplateMatcher, RejectsOccludedScene) {
  constexpr int W = 320;
  constexpr int H = 240;
  const cv::Rect r0(80, 60, 28, 24);

  const cv::Mat bg = makeBackground(W, H, 11);
  cv::Mat f0 = bg.clone();
  drawBlob(f0, r0);

  TemplateMatcher matcher(testConfig());
  matcher.reset(f0, r0);

  const auto m = matcher.track(bg, cv::Rect(30, 20, 220, 180));

  EXPECT_FALSE(m.valid) << "Background-only ROI must not be accepted as a reacquisition";
  EXPECT_LT(m.ncc, testConfig().min_ncc);
}

TEST(TemplateMatcher, ScoresCorrectCropBest) {
  constexpr int W = 320;
  constexpr int H = 240;
  const cv::Rect r0(80, 60, 28, 24);

  const cv::Mat bg = makeBackground(W, H, 13);
  cv::Mat f0 = bg.clone();
  drawBlob(f0, r0);

  TemplateMatcher matcher(testConfig());
  matcher.reset(f0, r0);

  const double on_target = matcher.scoreAt(f0, r0);
  const double off_target = matcher.scoreAt(f0, cv::Rect(r0.x + 12, r0.y + 9, r0.width, r0.height));
  EXPECT_GT(on_target, off_target) << "Exact crop should score higher than an offset crop";
  EXPECT_GT(on_target, testConfig().min_ncc);
}

TEST(TemplateMatcher, UpdateBlendsNewAppearance) {
  constexpr int W = 320;
  constexpr int H = 240;
  const cv::Rect r0(80, 60, 28, 24);

  const cv::Mat bg = makeBackground(W, H, 17);
  cv::Mat f0 = bg.clone();
  drawBlob(f0, r0);
  cv::Mat f1 = bg.clone();
  drawBlob(f1, r0);
  // Make the second appearance clearly different: invert the blob polarity.
  cv::Mat f1_inv = 255 - f1;

  TemplateMatcher matcher(testConfig());
  ASSERT_FALSE(matcher.hasTemplate());

  matcher.update(f0, r0, 1.0);  // alpha=1 overrides -> acts like reset
  ASSERT_TRUE(matcher.hasTemplate());
  const double before = matcher.scoreAt(f1_inv, r0);  // new appearance, old template

  matcher.update(f1_inv, r0, 1.0);  // re-learn the new appearance
  const double after = matcher.scoreAt(f1_inv, r0);

  EXPECT_GT(after, before) << "Template should track the updated appearance";
  EXPECT_GT(after, testConfig().min_ncc);
}