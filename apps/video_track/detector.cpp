#include "detector.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <sstream>

#include "kalman/core/profiling.hpp"

// ===========================================================================
//  YOLODetector
// ===========================================================================

YOLODetector::YOLODetector(const std::string& model_path, float conf_threshold, float nms_threshold)
    : conf_threshold_(conf_threshold), nms_threshold_(nms_threshold) {
  net_ = cv::dnn::readNet(model_path);
  if (net_.empty()) {
    std::fprintf(stderr, "Failed to load ONNX model: %s\n", model_path.c_str());
    std::fprintf(stderr, "Falling back to CPU backend.\n");
  }
  net_.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
  net_.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
}

std::optional<Detection> YOLODetector::next(const cv::Mat& frame) {
  auto dets = detect(frame);
  if (dets.empty()) return std::nullopt;
  return dets.front();
}

std::vector<Detection> YOLODetector::detect(const cv::Mat& frame) {
  KALMAN_PROFILE_ZONE_NAMED_C("YOLO::detect", kalman::profiling::ColorVision);
  // -----------------------------------------------------------------------
  //  1. Pre-process: blob from image
  // -----------------------------------------------------------------------
  cv::Mat blob;
  {
    KALMAN_PROFILE_ZONE_NAMED_C("YOLO::blob", kalman::profiling::ColorVision);
    blob = cv::dnn::blobFromImage(frame, 1.0f / 255.0f, cv::Size(input_width_, input_height_),
                                  cv::Scalar(), true, false);
  }
  net_.setInput(blob);

  // -----------------------------------------------------------------------
  //  2. Forward pass
  // -----------------------------------------------------------------------
  std::vector<cv::Mat> outputs;
  {
    KALMAN_PROFILE_ZONE_NAMED_C("YOLO::forward", kalman::profiling::ColorVision);
    net_.forward(outputs, net_.getUnconnectedOutLayersNames());
  }

  // YOLOv8n output shape: (1, 84, 8400)  — 84 = 4 bbox + 80 COCO classes
  const float* data = reinterpret_cast<const float*>(outputs[0].data);
  const int num_detections = outputs[0].size[2];  // 8400
  const int num_channels = outputs[0].size[1];    // 84

  float scale_x = static_cast<float>(frame.cols) / input_width_;
  float scale_y = static_cast<float>(frame.rows) / input_height_;

  // -----------------------------------------------------------------------
  //  3. Parse detections
  // -----------------------------------------------------------------------
  std::vector<cv::Rect> boxes;
  std::vector<float> scores;
  std::vector<int> class_ids;
  std::vector<int> indices;

  {
    KALMAN_PROFILE_ZONE_NAMED_C("YOLO::parse", kalman::profiling::ColorVision);
    for (int i = 0; i < num_detections; ++i) {
      // Find the max class score among COCO classes
      float max_score = 0.0f;
      int best_class = -1;
      for (int j = 4; j < num_channels; ++j) {
        float s = data[j * num_detections + i];
        if (s > max_score) {
          max_score = s;
          best_class = j - 4;
        }
      }

      // Filter by confidence
      if (max_score < conf_threshold_) continue;

      // Filter by target classes
      if (std::find(target_classes_.begin(), target_classes_.end(), best_class) ==
          target_classes_.end())
        continue;

      // Decode bbox (cx, cy, w, h) → (x1, y1, x2, y2)
      float cx = data[0 * num_detections + i];
      float cy = data[1 * num_detections + i];
      float w = data[2 * num_detections + i];
      float h = data[3 * num_detections + i];

      // Scale from network input size to original frame size
      cx *= scale_x;
      cy *= scale_y;
      w *= scale_x;
      h *= scale_y;

      float x1 = cx - w / 2.0f;
      float y1 = cy - h / 2.0f;
      float x2 = cx + w / 2.0f;
      float y2 = cy + h / 2.0f;

      boxes.emplace_back(static_cast<int>(x1), static_cast<int>(y1), static_cast<int>(x2 - x1),
                         static_cast<int>(y2 - y1));
      scores.push_back(max_score);
      class_ids.push_back(best_class);
    }
  }

  // -----------------------------------------------------------------------
  //  4. Non-maximum suppression
  // -----------------------------------------------------------------------
  auto keep_idx = nonMaxSuppression(boxes, scores, nms_threshold_);

  std::vector<Detection> results;
  results.reserve(keep_idx.size());
  for (int idx : keep_idx) {
    results.push_back({boxes[idx], scores[idx], class_ids[idx]});
  }

  // Sort by confidence descending
  std::sort(results.begin(), results.end(),
            [](const Detection& a, const Detection& b) { return a.confidence > b.confidence; });

  return results;
}

std::vector<int> YOLODetector::nonMaxSuppression(const std::vector<cv::Rect>& boxes,
                                                 const std::vector<float>& scores,
                                                 float nms_threshold) const {
  KALMAN_PROFILE_ZONE_NAMED_C("YOLO::nms", kalman::profiling::ColorVision);
  std::vector<int> indices;

  std::vector<int> order(static_cast<int>(boxes.size()));
  for (int i = 0; i < static_cast<int>(boxes.size()); ++i) order[i] = i;
  std::sort(order.begin(), order.end(), [&](int a, int b) { return scores[a] > scores[b]; });

  std::vector<bool> suppressed(boxes.size(), false);
  for (int i = 0; i < static_cast<int>(order.size()); ++i) {
    if (suppressed[order[i]]) continue;
    indices.push_back(order[i]);
    for (int j = i + 1; j < static_cast<int>(order.size()); ++j) {
      if (suppressed[order[j]]) continue;
      float iou = static_cast<float>((boxes[order[i]] & boxes[order[j]]).area()) /
                  static_cast<float>((boxes[order[i]] | boxes[order[j]]).area());
      if (iou > nms_threshold) suppressed[order[j]] = true;
    }
  }

  return indices;
}

// ===========================================================================
//  CSVDetectionSource
// ===========================================================================

CSVDetectionSource::CSVDetectionSource(const std::string& csv_path) {
  std::ifstream file(csv_path);
  if (!file.is_open()) {
    std::fprintf(stderr, "Warning: could not open detections CSV: %s\n", csv_path.c_str());
    return;
  }

  std::string line;
  // Skip header
  std::getline(file, line);

  int max_frame = 0;
  while (std::getline(file, line)) {
    std::istringstream ss(line);
    std::string token;

    if (!std::getline(ss, token, ',')) continue;
    int frame_idx = std::stoi(token);

    if (!std::getline(ss, token, ',')) continue;
    float cx = std::stof(token);

    if (!std::getline(ss, token, ',')) continue;
    float cy = std::stof(token);

    // Pad with nullopt up to this frame
    while (static_cast<int>(detections_.size()) <= frame_idx) {
      detections_.push_back(std::nullopt);
    }
    // CSV has no bbox; create a small default bbox centered on the point
    const int half_size = 15;
    cv::Rect bbox(static_cast<int>(cx) - half_size, static_cast<int>(cy) - half_size, 2 * half_size,
                  2 * half_size);
    detections_[frame_idx] = Detection{bbox, 1.0f, -1};
    max_frame = std::max(max_frame, frame_idx);
  }
}

std::optional<Detection> CSVDetectionSource::next(const cv::Mat&) {
  if (frame_idx_ >= detections_.size()) return std::nullopt;
  return detections_[frame_idx_++];
}
