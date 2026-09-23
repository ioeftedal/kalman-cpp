#pragma once

#include <cstdint>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <vector>

// ---------------------------------------------------------------------------
//  Lightweight appearance memory (multi-scale NCC template matcher)
//
//  Stores a small grayscale patch of the target and re-localises it inside a
//  search ROI with TM_CCOEFF_NORMED across a modest scale sweep. Used to
//  (a) reacquire the target while the detector drops out or a detection is
//  gated out, and (b) score a candidate detection crop for gate overrides.
//  This is the classical (correlation-filter) analogue of the appearance +
//  memory heads of modern SAMURAI-style trackers, at negligible CPU cost.
// ---------------------------------------------------------------------------

namespace kalman::tracking {

struct AppearanceConfig {
  int template_size = 40;     // fixed template grid edge (px)
  double pad_scale = 0.35;    // extra context around the bbox when sampling a patch
  double min_ncc = 0.78;      // minimum TM_CCOEFF_NORMED peak to accept a match
  double min_psr = 3.0;       // minimum peak-to-sidelobe ratio to accept a match
  double update_rate = 0.15;  // template update gain in (0,1]; 0 = frozen template
  double scale_step = 0.125;  // relative-scale sweep step
  int scale_steps = 3;        // number of sweep steps above/below 1.0
};

struct AppearanceMatch {
  bool valid = false;
  cv::Point2f position;  // matched centre in full-frame coordinates
  cv::Rect bbox;         // matched bbox (full-frame coordinates)
  double ncc = 0.0;      // correlation peak (TM_CCOEFF_NORMED)
  double psr = 0.0;      // peak-to-sidelobe ratio
  double scale = 1.0;    // relative target scale vs. template nominal
};

class TemplateMatcher {
 public:
  explicit TemplateMatcher(AppearanceConfig cfg = {});

  /// (Re)initialise the template from a patch centred on `bbox` of `frame`.
  void reset(const cv::Mat& frame, const cv::Rect& bbox);

  /// Blend (1-alpha)*old + alpha*new of the current patch. `alpha <= 0` uses
  /// cfg.update_rate. Call only with high-confidence target locations.
  void update(const cv::Mat& frame, const cv::Rect& bbox, double alpha = -1.0);

  bool hasTemplate() const { return !template_.empty(); }

  /// Multi-scale NCC search inside `roi` (full-frame coordinates).
  /// The returned centre is the reacquired target position.
  AppearanceMatch track(const cv::Mat& frame, const cv::Rect& roi) const;

  /// NCC similarity of the template against a specific bbox crop.
  double scoreAt(const cv::Mat& frame, const cv::Rect& bbox) const;

  double lastNominalSize() const { return nominal_bbox_side_; }

 private:
  cv::Rect padded(const cv::Rect& bbox) const;
  int sampleSizeFor(const cv::Rect& bbox) const;
  cv::Mat samplePatch(const cv::Mat& frame, const cv::Rect& bbox, int out) const;
  std::vector<double> scales() const;

  AppearanceConfig cfg_;
  cv::Mat template_;           // grayscale template at cfg_.template_size grid
  int nominal_bbox_side_ = 0;  // target bbox side (px) the template was built from
};

}  // namespace kalman::tracking