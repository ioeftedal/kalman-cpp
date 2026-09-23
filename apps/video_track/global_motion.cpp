#include "global_motion.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <opencv2/calib3d.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/video.hpp>

#include "kalman/core/profiling.hpp"

GlobalMotionEstimator::GlobalMotionEstimator() : GlobalMotionEstimator(Config{}) {}
GlobalMotionEstimator::GlobalMotionEstimator(Config cfg) : cfg_(cfg) {}

void GlobalMotionEstimator::detectCorners(const cv::Mat& gray, const cv::Rect& excluded,
                                          std::vector<cv::Point2f>& pts) {
  cv::Mat mask = cv::Mat::ones(gray.size(), CV_8UC1);
  cv::Rect full(0, 0, gray.cols, gray.rows);
  cv::Rect exclude = full & excluded;
  if (!exclude.empty()) {
    mask(exclude) = 0;
  }
  cv::goodFeaturesToTrack(gray, pts, cfg_.max_features, cfg_.quality_level, cfg_.min_distance, mask,
                          cfg_.block_size, true, 0.04);
}

GlobalMotion GlobalMotionEstimator::track(const cv::Mat& frame, const cv::Rect& exclude_roi) {
  KALMAN_PROFILE_ZONE_NAMED_C("GlobalMotion::track", kalman::profiling::ColorMotion);
  GlobalMotion out;

  cv::Mat gray;
  if (frame.channels() == 3) {
    cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
  } else {
    gray = frame;
  }

  cv::Mat work;
  if (cfg_.downsample < 1.0) {
    cv::resize(gray, work, cv::Size(), cfg_.downsample, cfg_.downsample, cv::INTER_AREA);
  } else {
    work = gray;
  }

  cv::Rect excluded(static_cast<int>(exclude_roi.x * cfg_.downsample),
                    static_cast<int>(exclude_roi.y * cfg_.downsample),
                    static_cast<int>(exclude_roi.width * cfg_.downsample),
                    static_cast<int>(exclude_roi.height * cfg_.downsample));

  if (!have_prev_) {
    detectCorners(work, excluded, prev_pts_);
    prev_gray_ = work;
    have_prev_ = true;
    return out;
  }

  // Corner detection can legitimately yield nothing (e.g. the excluded search
  // ROI covers the whole frame right after initialisation). LK would reject an
  // empty point set, so re-detect instead of passing zero points forward.
  if (prev_pts_.empty()) {
    detectCorners(work, excluded, prev_pts_);
    prev_gray_ = work;
    return out;
  }

  std::vector<cv::Point2f> curr_pts(prev_pts_.size());
  std::vector<uchar> status;
  std::vector<float> err;
  cv::calcOpticalFlowPyrLK(prev_gray_, work, prev_pts_, curr_pts, status, err,
                           cv::Size(cfg_.win_size, cfg_.win_size), cfg_.max_level);

  std::vector<cv::Point2f> prev_in, curr_in;
  std::vector<cv::Point2f> tracked;
  prev_in.reserve(prev_pts_.size());
  curr_in.reserve(prev_pts_.size());
  tracked.reserve(prev_pts_.size());
  for (size_t i = 0; i < prev_pts_.size(); ++i) {
    if (!status[i]) continue;
    if (excluded.contains(prev_pts_[i])) continue;
    prev_in.push_back(prev_pts_[i]);
    curr_in.push_back(curr_pts[i]);
    tracked.push_back(curr_pts[i]);
  }

  if (static_cast<int>(prev_in.size()) < cfg_.min_inliers) {
    prev_gray_ = work;
    prev_pts_ = std::move(tracked);
    return out;
  }

  cv::Mat inlier_mask;
  cv::estimateAffinePartial2D(prev_in, curr_in, inlier_mask, cv::RANSAC, cfg_.ransac_threshold);

  int inlier_count = 0;
  for (int i = 0; i < inlier_mask.rows; ++i) {
    if (inlier_mask.at<uchar>(i)) inlier_count++;
  }

  const double ratio =
      static_cast<double>(inlier_count) / static_cast<double>(std::max<size_t>(1, prev_in.size()));
  prev_gray_ = work;
  prev_pts_ = std::move(tracked);

  if (inlier_count < cfg_.min_inliers || ratio < cfg_.min_inlier_ratio) {
    return out;
  }

  cv::Point2f sum{0, 0};
  for (int i = 0; i < inlier_mask.rows; ++i) {
    if (inlier_mask.at<uchar>(i)) {
      sum += curr_in[i] - prev_in[i];
    }
  }
  sum *= 1.0f / static_cast<float>(inlier_count);

  out.displacement = sum / static_cast<float>(cfg_.downsample);
  out.inliers = inlier_count;
  out.inlier_ratio = ratio;
  out.valid = true;
  return out;
}

void GlobalMotionEstimator::reset() {
  prev_pts_.clear();
  prev_gray_ = cv::Mat();
  have_prev_ = false;
}