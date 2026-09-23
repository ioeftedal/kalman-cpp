/// @file
/// @brief MixFormerV2-S single-object tracker backend on ONNX Runtime.
///
/// One-stream transformer tracker (NeurIPS 2023): state of the art in the
/// "first real-time-on-CPU transformer tracker" class (GOT-10k / LaSOT).
/// Runs a 224x224 search patch vs. (static + online) 112x112 templates each
/// frame. Initialised from an object-detector box, it tracks any appearance
/// and therefore generalises across videos and live feeds without per-video
/// tuning.

#pragma once

#include <memory>
#include <opencv2/core.hpp>
#include <optional>
#include <string>

struct DeepTrackResult {
  cv::Rect bbox;       ///< tracked target box (top-left, w, h) in frame coords
  float score = 0.0f;  ///< tracker confidence in [0, 1]
  bool valid = false;
};

/// ONNX Runtime wrapper for the MixFormerV2-S tracking network. The model
/// stays behind a pimpl so ONNX Runtime headers never leak into consumers.
class DeepTracker {
 public:
  DeepTracker();
  ~DeepTracker();

  DeepTracker(const DeepTracker&) = delete;
  DeepTracker& operator=(const DeepTracker&) = delete;

  /// Load an ONNX model with inputs img_t/img_ot/img_search and outputs
  /// pred_boxes/pred_scores. Returns false if the graph shape is unexpected.
  bool load(const std::string& model_path);

  [[nodiscard]] bool loaded() const;

  /// Initialise the tracker from a target box (top-left, w, h) in frame px.
  bool init(const cv::Mat& frame, const cv::Rect& bbox);

  /// Track the target in the next frame.
  DeepTrackResult track(const cv::Mat& frame);

  void reset();

  /// Mean per-frame inference wall time (ms) over the last call.
  [[nodiscard]] double lastInferenceMs() const { return last_ms_; }

  /// Number of online-template updates committed so far.
  [[nodiscard]] int templateUpdates() const { return template_updates_; }

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
  double last_ms_ = 0.0;
  int template_updates_ = 0;
};