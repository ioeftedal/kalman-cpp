/// @file
/// @brief Camera-motion estimator: robust background translation for
///        moving-camera drone footage.

#pragma once

#include <opencv2/core.hpp>
#include <vector>

struct GlobalMotion {
  bool valid = false;
  cv::Point2f displacement{0, 0};
  int inliers = 0;
  double inlier_ratio = 0.0;
};

/// Estimates the inter-frame background shift caused by camera motion so the
/// target's optical-flow velocity can be compensated before it feeds the UKF.
/// Background features are tracked with pyramidal LK and the shift is fitted
/// robustly with RANSAC (affine, partial: translation/rotation/uniform scale).
class GlobalMotionEstimator {
 public:
  struct Config {
    int max_features = 200;
    double quality_level = 0.01;
    double min_distance = 10.0;
    int block_size = 5;
    int win_size = 21;
    int max_level = 3;
    double downsample = 0.5;
    double ransac_threshold = 3.0;
    double min_inlier_ratio = 0.4;
    int min_inliers = 8;
  };

  GlobalMotionEstimator();
  explicit GlobalMotionEstimator(Config cfg);

  /// Process one frame; `exclude_roi` (full-res) is masked out so features on
  /// the tracked target do not bias the background estimate.
  GlobalMotion track(const cv::Mat& frame, const cv::Rect& exclude_roi);

  void reset();

 private:
  void detectCorners(const cv::Mat& gray, const cv::Rect& excluded, std::vector<cv::Point2f>& pts);

  Config cfg_;
  cv::Mat prev_gray_;
  std::vector<cv::Point2f> prev_pts_;
  bool have_prev_ = false;
};