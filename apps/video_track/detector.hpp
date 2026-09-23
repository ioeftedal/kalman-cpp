#pragma once

#include <opencv2/core.hpp>
#include <opencv2/dnn.hpp>
#include <optional>
#include <string>
#include <vector>

// ---------------------------------------------------------------------------
//  Detection data
// ---------------------------------------------------------------------------

struct Detection {
  cv::Rect bbox;
  float confidence;
  int class_id;

  [[nodiscard]] cv::Point2f center() const {
    return cv::Point2f(bbox.x + bbox.width / 2.0f, bbox.y + bbox.height / 2.0f);
  }
};

// ---------------------------------------------------------------------------
//  Abstract detection source — allows live YOLO or pre-computed CSV
// ---------------------------------------------------------------------------

class DetectionSource {
 public:
  virtual ~DetectionSource() = default;
  virtual std::optional<Detection> next(const cv::Mat& frame) = 0;
};

// ---------------------------------------------------------------------------
//  YOLOv8 detector (ONNX via OpenCV DNN)
// ---------------------------------------------------------------------------

class YOLODetector : public DetectionSource {
 public:
  YOLODetector(const std::string& model_path, float conf_threshold = 0.5f,
               float nms_threshold = 0.45f, int input_size = 640, bool letterbox = true);

  std::optional<Detection> next(const cv::Mat& frame) override;

  /// Return all detections for a frame (useful for visualisation).
  std::vector<Detection> detect(const cv::Mat& frame);

 private:
  std::vector<int> nonMaxSuppression(const std::vector<cv::Rect>& boxes,
                                     const std::vector<float>& scores, float nms_threshold) const;

  cv::dnn::Net net_;
  float conf_threshold_;
  float nms_threshold_;
  int input_width_ = 640;
  int input_height_ = 640;
  bool letterbox_ = true;
  float scale_ = 1.0f;                         // letterbox scale (input px per frame px)
  int pad_x_ = 0;                              // letterbox x offset (input coords)
  int pad_y_ = 0;                              // letterbox y offset (input coords)
  std::vector<int> target_classes_ = {4, 14};  // COCO: airplane, bird
};

// ---------------------------------------------------------------------------
//  Pre-computed detection CSV reader (fallback / test mode)
// ---------------------------------------------------------------------------

class CSVDetectionSource : public DetectionSource {
 public:
  explicit CSVDetectionSource(const std::string& csv_path);
  std::optional<Detection> next(const cv::Mat& frame) override;

 private:
  std::vector<std::optional<Detection>> detections_;
  size_t frame_idx_ = 0;
};
