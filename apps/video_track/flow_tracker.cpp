#include "flow_tracker.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>

FlowTracker::FlowTracker() : cfg_(Config{}) {}
FlowTracker::FlowTracker(Config cfg) : cfg_(cfg) {}

FlowResult FlowTracker::track(const cv::Mat& frame, const cv::Rect& search_roi) {
  FlowResult result;

  if (prev_pts_.empty() || !has_prev_frame_) {
    return result;
  }

  cv::Mat curr_gray;
  if (frame.channels() == 3) {
    cv::cvtColor(frame, curr_gray, cv::COLOR_BGR2GRAY);
  } else {
    curr_gray = frame;
  }

  // Forward tracking: frame_{t-1} → frame_t
  std::vector<cv::Point2f> next_pts(prev_pts_.size());
  std::vector<uchar> fwd_status;
  std::vector<float> fwd_err;
  cv::calcOpticalFlowPyrLK(prev_gray_, curr_gray, prev_pts_, next_pts, fwd_status, fwd_err,
                           cv::Size(cfg_.win_size, cfg_.win_size), cfg_.max_level);

  // Backward tracking: frame_t → frame_{t-1} (for forward-backward check)
  // Pre-allocate; OpenCV will copy next_pts as the initial estimate (no flag needed)
  std::vector<cv::Point2f> back_pts(next_pts.size());
  std::vector<uchar> back_status;
  std::vector<float> back_err;
  cv::calcOpticalFlowPyrLK(
      curr_gray, prev_gray_, next_pts, back_pts, back_status, back_err,
      cv::Size(cfg_.win_size, cfg_.win_size), cfg_.max_level,
      cv::TermCriteria(cv::TermCriteria::COUNT + cv::TermCriteria::EPS, 30, 0.01));

  // Expanded ROI margin (features near the edge are likely still valid)
  cv::Rect roi_margin(search_roi.x - search_roi.width / 2, search_roi.y - search_roi.height / 2,
                      search_roi.width * 2, search_roi.height * 2);

  // Collect valid features: passed forward+backward tracking, inside ROI
  std::vector<cv::Point2f> valid_next;
  std::vector<cv::Point2f> displacements;
  std::vector<float> feature_errors;
  valid_next.reserve(prev_pts_.size());
  displacements.reserve(prev_pts_.size());
  feature_errors.reserve(prev_pts_.size());

  for (size_t i = 0; i < prev_pts_.size(); ++i) {
    if (!fwd_status[i] || !back_status[i]) continue;

    float fb_err = std::sqrt(std::pow(next_pts[i].x - back_pts[i].x, 2) +
                             std::pow(next_pts[i].y - back_pts[i].y, 2));
    if (fb_err > cfg_.fb_threshold) continue;

    if (!roi_margin.contains(next_pts[i])) continue;

    valid_next.push_back(next_pts[i]);
    displacements.push_back(next_pts[i] - prev_pts_[i]);
    feature_errors.push_back(fb_err);
  }

  if (static_cast<int>(displacements.size()) < cfg_.min_features) {
    return result;
  }

  // Median displacement (robust against outliers)
  std::vector<float> dx_vals, dy_vals;
  dx_vals.reserve(displacements.size());
  dy_vals.reserve(displacements.size());
  for (const auto& d : displacements) {
    dx_vals.push_back(d.x);
    dy_vals.push_back(d.y);
  }

  auto computeMedian = [](std::vector<float>& vals) -> float {
    size_t mid = vals.size() / 2;
    std::nth_element(vals.begin(), vals.begin() + static_cast<long>(mid), vals.end());
    return vals[mid];
  };

  float med_dx = computeMedian(dx_vals);
  float med_dy = computeMedian(dy_vals);

  // Residual filtering: reject displacement vectors far from median
  std::vector<cv::Point2f> filtered_disp;
  std::vector<cv::Point2f> filtered_pts;
  std::vector<float> filtered_errors;
  filtered_disp.reserve(displacements.size());
  filtered_pts.reserve(displacements.size());
  filtered_errors.reserve(displacements.size());

  for (size_t i = 0; i < displacements.size(); ++i) {
    float res = std::sqrt(std::pow(displacements[i].x - med_dx, 2) +
                          std::pow(displacements[i].y - med_dy, 2));
    if (res < cfg_.resid_threshold) {
      filtered_disp.push_back(displacements[i]);
      filtered_pts.push_back(valid_next[i]);
      filtered_errors.push_back(feature_errors[i]);
    }
  }

  if (static_cast<int>(filtered_disp.size()) < cfg_.min_features) {
    return result;
  }

  // Mean displacement from survivors
  double mx = 0.0, my = 0.0;
  for (const auto& d : filtered_disp) {
    mx += d.x;
    my += d.y;
  }
  mx /= static_cast<double>(filtered_disp.size());
  my /= static_cast<double>(filtered_disp.size());

  // Sample variance → displacement uncertainty
  double vx = 0.0, vy = 0.0;
  for (const auto& d : filtered_disp) {
    vx += std::pow(d.x - mx, 2);
    vy += std::pow(d.y - my, 2);
  }
  vx /= static_cast<double>(filtered_disp.size() - 1);
  vy /= static_cast<double>(filtered_disp.size() - 1);

  double sigma_d =
      std::max(0.3, std::sqrt((vx + vy) / (2.0 * static_cast<double>(filtered_disp.size()))));

  // Update internal state for the next frame
  prev_gray_ = curr_gray;
  prev_pts_ = filtered_pts;

  result.valid = true;
  result.mean_displacement = cv::Point2f(static_cast<float>(mx), static_cast<float>(my));
  result.uncertainty = sigma_d;
  result.num_features = static_cast<int>(filtered_disp.size());
  result.per_feature_error = filtered_errors;
  result.feature_points = filtered_pts;
  return result;
}

void FlowTracker::refreshFeatures(const cv::Mat& frame, const cv::Rect& bbox) {
  cv::Mat gray;
  if (frame.channels() == 3) {
    cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
  } else {
    gray = frame;
  }

  // Slightly expand the bbox to include border context for tracking
  int margin_x = static_cast<int>(bbox.width * 0.2);
  int margin_y = static_cast<int>(bbox.height * 0.2);
  cv::Rect roi(std::max(0, bbox.x - margin_x), std::max(0, bbox.y - margin_y),
               std::min(frame.cols - bbox.x + margin_x, bbox.width + 2 * margin_x),
               std::min(frame.rows - bbox.y + margin_y, bbox.height + 2 * margin_y));

  // Mask: only detect corners within the expanded ROI
  cv::Mat mask = cv::Mat::zeros(gray.size(), CV_8UC1);
  mask(roi) = 255;

  prev_pts_.clear();
  cv::goodFeaturesToTrack(gray, prev_pts_, cfg_.max_corners, cfg_.quality_level, cfg_.min_distance,
                          mask, cfg_.block_size, true, 0.04);

  prev_gray_ = gray;
  has_prev_frame_ = true;
}
