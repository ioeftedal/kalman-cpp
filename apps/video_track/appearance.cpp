#include "appearance.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

namespace kalman::tracking {

namespace {

double computePSR(const cv::Mat& response, const cv::Point& peak, int halo) {
  cv::Mat mask = cv::Mat::zeros(response.size(), CV_8U);
  cv::Rect excl(std::max(0, peak.x - halo), std::max(0, peak.y - halo),
                std::min(response.cols, peak.x + halo + 1) - std::max(0, peak.x - halo),
                std::min(response.rows, peak.y + halo + 1) - std::max(0, peak.y - halo));
  cv::rectangle(mask, excl, cv::Scalar(255), cv::FILLED);
  cv::Scalar m;
  cv::Scalar s;
  cv::meanStdDev(response, m, s, ~mask);
  const double mean = m[0];
  const double stddev = s[0];
  if (stddev < 1e-8) {
    return 0.0;
  }
  return (response.at<float>(peak) - mean) / stddev;
}

}  // namespace

TemplateMatcher::TemplateMatcher(AppearanceConfig cfg) : cfg_(cfg) {}

cv::Rect TemplateMatcher::padded(const cv::Rect& bbox) const {
  const int dx = static_cast<int>(std::round(bbox.width * cfg_.pad_scale));
  const int dy = static_cast<int>(std::round(bbox.height * cfg_.pad_scale));
  return bbox + cv::Size(2 * dx, 2 * dy) - cv::Point(dx, dy);
}

int TemplateMatcher::sampleSizeFor(const cv::Rect& bbox) const {
  const int side = static_cast<int>(
      std::round(std::max(bbox.width, bbox.height) * (1.0 + (2.0 * cfg_.pad_scale))));
  return std::max(8, side);
}

cv::Mat TemplateMatcher::samplePatch(const cv::Mat& frame, const cv::Rect& bbox, int out) const {
  cv::Mat gray;
  if (frame.channels() == 3) {
    cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
  } else if (frame.channels() == 1) {
    gray = frame;
  } else {
    cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
  }

  const cv::Rect p = padded(bbox);
  const cv::Rect clamped = p & cv::Rect(0, 0, gray.cols, gray.rows);
  if (clamped.width <= 1 || clamped.height <= 1) {
    return {};
  }

  cv::Mat patch = gray(clamped);
  cv::resize(patch, patch, cv::Size(out, out), 0.0, 0.0, cv::INTER_LINEAR);
  if (patch.empty()) {
    return {};
  }
  patch.convertTo(patch, CV_32F);
  cv::subtract(patch, cv::mean(patch)[0], patch);
  patch.convertTo(patch, CV_8U, 1.0, 128.0);
  return patch;
}

void TemplateMatcher::reset(const cv::Mat& frame, const cv::Rect& bbox) {
  cv::Mat patch = samplePatch(frame, bbox, cfg_.template_size);
  if (patch.empty()) {
    template_.release();
    nominal_bbox_side_ = 0;
    return;
  }
  template_ = patch.clone();
  nominal_bbox_side_ = static_cast<int>(std::round(std::max(bbox.width, bbox.height)));
}

void TemplateMatcher::update(const cv::Mat& frame, const cv::Rect& bbox, double alpha) {
  if (template_.empty()) {
    reset(frame, bbox);
    return;
  }
  cv::Mat patch = samplePatch(frame, bbox, cfg_.template_size);
  if (patch.empty()) {
    return;
  }

  const double lr = (alpha > 0.0) ? std::min(1.0, alpha) : cfg_.update_rate;
  const int side = static_cast<int>(std::round(std::max(bbox.width, bbox.height)));
  if (lr < 1.0) {
    cv::addWeighted(template_, 1.0 - lr, patch, lr, 0.0, template_);
  } else {
    template_ = patch.clone();
  }
  nominal_bbox_side_ = static_cast<int>(std::round((0.9 * nominal_bbox_side_) + (0.1 * side)));
}

std::vector<double> TemplateMatcher::scales() const {
  std::vector<double> s;
  for (int i = -cfg_.scale_steps; i <= cfg_.scale_steps; ++i) {
    s.push_back(1.0 + (i * cfg_.scale_step));
  }
  return s;
}

double TemplateMatcher::scoreAt(const cv::Mat& frame, const cv::Rect& bbox) const {
  if (template_.empty()) {
    return 0.0;
  }
  cv::Mat patch = samplePatch(frame, bbox, cfg_.template_size);
  if (patch.empty()) {
    return 0.0;
  }
  cv::Mat res;
  cv::matchTemplate(patch, template_, res, cv::TM_CCOEFF_NORMED);
  return static_cast<double>(res.at<float>(0, 0));
}

AppearanceMatch TemplateMatcher::track(const cv::Mat& frame, const cv::Rect& roi) const {
  AppearanceMatch best;
  if (template_.empty() || roi.width <= 0 || roi.height <= 0) {
    return best;
  }

  cv::Mat gray;
  if (frame.channels() == 3) {
    cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
  } else if (frame.channels() == 1) {
    gray = frame;
  } else {
    cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
  }

  const cv::Rect src_roi = roi & cv::Rect(0, 0, gray.cols, gray.rows);
  if (src_roi.width < 8 || src_roi.height < 8) {
    return best;
  }

  const double min_side = 8.0;
  for (const double s : scales()) {
    const int tpl_w = static_cast<int>(std::round(cfg_.template_size * s));
    const int tpl_h = tpl_w;
    if (tpl_w > src_roi.width || tpl_h > src_roi.height) {
      continue;
    }

    const int phys_side = static_cast<int>(std::round(cfg_.template_size * s * nominal_bbox_side_ /
                                                      static_cast<double>(cfg_.template_size)));
    if (static_cast<double>(phys_side) < min_side) {
      continue;
    }

    cv::Mat tpl;
    cv::resize(template_, tpl, cv::Size(tpl_w, tpl_h), 0.0, 0.0, cv::INTER_LINEAR);

    cv::Mat response;
    cv::matchTemplate(gray(src_roi), tpl, response, cv::TM_CCOEFF_NORMED);
    if (response.empty()) {
      continue;
    }

    double peak = -1.0;
    cv::Point loc;
    cv::minMaxLoc(response, nullptr, &peak, nullptr, &loc);
    const double psr = computePSR(response, loc, 5);

    if (peak > best.ncc) {
      best.ncc = peak;
      best.psr = psr;
      best.scale = s;
      const double cx = src_roi.x + loc.x + (tpl_w / 2.0);
      const double cy = src_roi.y + loc.y + (tpl_h / 2.0);
      const int side = static_cast<int>(std::round(s * nominal_bbox_side_));
      best.position = cv::Point2f(static_cast<float>(cx), static_cast<float>(cy));
      best.bbox = cv::Rect(static_cast<int>(std::round(cx - (side / 2.0))),
                           static_cast<int>(std::round(cy - (side / 2.0))), side, side);
    }
  }

  if (best.ncc >= cfg_.min_ncc && best.psr >= cfg_.min_psr) {
    best.valid = true;
  }
  return best;
}

}  // namespace kalman::tracking