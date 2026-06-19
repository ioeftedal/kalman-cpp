#pragma once

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/video.hpp>
#include <vector>

struct FlowResult {
  bool valid = false;
  cv::Point2f mean_displacement{0, 0};
  double uncertainty = 0.0;
  int num_features = 0;
  std::vector<float> per_feature_error;     // forward-backward error per tracked feature
  std::vector<cv::Point2f> feature_points;  // current image coordinates of tracked features
};

class FlowTracker {
 public:
  struct Config {
    int max_corners = 25;
    double quality_level = 0.02;
    double min_distance = 5.0;
    int block_size = 7;
    int win_size = 21;
    int max_level = 3;
    double fb_threshold = 1.5;
    double resid_threshold = 5.0;
    int min_features = 4;
  };

  FlowTracker();
  explicit FlowTracker(Config cfg);

  FlowResult track(const cv::Mat& frame, const cv::Rect& search_roi);

  void refreshFeatures(const cv::Mat& frame, const cv::Rect& bbox);

  bool hasFeatures() const { return !prev_pts_.empty(); }

  const std::vector<cv::Point2f>& getFeatures() const { return prev_pts_; }

 private:
  Config cfg_;
  std::vector<cv::Point2f> prev_pts_;
  cv::Mat prev_gray_;
  bool has_prev_frame_ = false;
};
