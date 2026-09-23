#include "deep_tracker.hpp"

#include <onnxruntime_cxx_api.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <opencv2/imgproc.hpp>
#include <vector>

namespace {

// ---------------------------------------------------------------------------
//  Mirror of the official run_mixformer2_onnx.py pre/post-processing exactly.
// ---------------------------------------------------------------------------

constexpr double kSearchFactor = 4.5;
constexpr int kSearchSize = 224;
constexpr double kTemplateFactor = 2.0;
constexpr int kTemplateSize = 112;
constexpr int kUpdateInterval = 200;
constexpr double kMaxScoreDecay = 1.0;
constexpr double kClipMargin = 10.0;
constexpr float kMean[3] = {0.485f, 0.456f, 0.406f};
constexpr float kStd[3] = {0.229f, 0.224f, 0.225f};

/// Square crop centred on the box with side sqrt(w*h)*area_factor, clipped to
/// the image, border-padded to a square and resized to `out_sz`. Identical to
/// `sample_target()` in the reference implementation.
cv::Mat sampleTarget(const cv::Mat& im, double x, double y, double w, double h, double area_factor,
                     int out_sz, double& resize_factor) {
  const int crop_sz = std::max(1, static_cast<int>(std::ceil(std::sqrt(w * h) * area_factor)));

  const int x1 = static_cast<int>(std::lround(x + 0.5 * w - 0.5 * crop_sz));
  const int x2 = x1 + crop_sz;
  const int y1 = static_cast<int>(std::lround(y + 0.5 * h - 0.5 * crop_sz));
  const int y2 = y1 + crop_sz;

  const int x1_pad = std::max(0, -x1);
  const int x2_pad = std::max(x2 - im.cols + 1, 0);
  const int y1_pad = std::max(0, -y1);
  const int y2_pad = std::max(y2 - im.rows + 1, 0);

  const int sx = x1 + x1_pad;
  const int sy = y1 + y1_pad;
  const int ex = x2 - x2_pad;
  const int ey = y2 - y2_pad;

  cv::Mat im_crop;
  if (ex > sx && ey > sy) {
    im_crop = im(cv::Rect(sx, sy, ex - sx, ey - sy)).clone();
  }
  cv::copyMakeBorder(im_crop, im_crop, y1_pad, y2_pad, x1_pad, x2_pad, cv::BORDER_CONSTANT);

  resize_factor = static_cast<double>(out_sz) / crop_sz;
  cv::resize(im_crop, im_crop, cv::Size(out_sz, out_sz), 0, 0, cv::INTER_LINEAR);
  return im_crop;
}

/// BGR u8 square patch -> flat CHW float tensor normalised like the official
/// `Preprocessor_wo_mask.process()`. Channel order is intentionally kept as
/// BGR to match the reference model export.
std::vector<float> preprocess(const cv::Mat& patch) {
  const int s = patch.cols;
  std::vector<float> buf(3u * static_cast<size_t>(s) * static_cast<size_t>(s));
  for (int y = 0; y < s; ++y) {
    const cv::Vec3b* p = patch.ptr<cv::Vec3b>(y);
    for (int x = 0; x < s; ++x) {
      for (int c = 0; c < 3; ++c) {
        buf[static_cast<size_t>(c) * static_cast<size_t>(s) * static_cast<size_t>(s) +
            static_cast<size_t>(y) * static_cast<size_t>(s) + static_cast<size_t>(x)] =
            (static_cast<float>(p[x][c]) / 255.0f - kMean[c]) / kStd[c];
      }
    }
  }
  return buf;
}

/// Replicates `clip_box()` of the reference implementation.
std::array<double, 4> clipBox(double x1, double y1, double w, double h, int H, int W) {
  const int margin = static_cast<int>(kClipMargin);
  const double max_x = std::max(0, W - margin);
  const double max_y = std::max(0, H - margin);
  double x2 = x1 + w;
  double y2 = y1 + h;
  x1 = std::min(std::max(0.0, x1), max_x);
  x2 = std::min(std::max(static_cast<double>(margin), x2), static_cast<double>(W));
  y1 = std::min(std::max(0.0, y1), max_y);
  y2 = std::min(std::max(static_cast<double>(margin), y2), static_cast<double>(H));
  w = std::max(static_cast<double>(margin), x2 - x1);
  h = std::max(static_cast<double>(margin), y2 - y1);
  return {x1, y1, w, h};
}

}  // namespace

// ---------------------------------------------------------------------------

struct DeepTracker::Impl {
  Ort::Env env{ORT_LOGGING_LEVEL_WARNING, "deep_tracker"};
  Ort::SessionOptions session_options;
  Ort::MemoryInfo mem_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeDefault);
  std::optional<Ort::Session> session;

  // Template state (flat CHW f32, BGR-normalised).
  std::vector<float> template_, online_template_, online_max_template_;

  // Current target box [x, y, w, h].
  std::array<double, 4> state{0, 0, 0, 0};
  int frame_id = 0;
  float max_pred_score = -1.0f;
};

DeepTracker::DeepTracker() : impl_(std::make_unique<Impl>()) {}
DeepTracker::~DeepTracker() = default;

bool DeepTracker::load(const std::string& model_path) {
  impl_->session_options.SetGraphOptimizationLevel(ORT_ENABLE_ALL);
  try {
    impl_->session.emplace(impl_->env, model_path.c_str(), impl_->session_options);
  } catch (const Ort::Exception& e) {
    std::fprintf(stderr, "DeepTracker: failed to load %s: %s\n", model_path.c_str(), e.what());
    impl_->session.reset();
    return false;
  }
  if (impl_->session->GetInputCount() != 3 || impl_->session->GetOutputCount() != 2) {
    std::fprintf(stderr, "DeepTracker: unexpected graph I/O (%zu in, %zu out)\n",
                 impl_->session->GetInputCount(), impl_->session->GetOutputCount());
    impl_->session.reset();
    return false;
  }
  std::printf("DeepTracker: loaded %s (MixFormerV2-S, CPU)\n", model_path.c_str());
  return true;
}

bool DeepTracker::loaded() const { return impl_->session.has_value(); }

bool DeepTracker::init(const cv::Mat& frame, const cv::Rect& bbox) {
  if (!loaded() || frame.empty()) return false;
  impl_->state = {static_cast<double>(bbox.x), static_cast<double>(bbox.y),
                  static_cast<double>(bbox.width), static_cast<double>(bbox.height)};
  double resize_factor = 0.0;
  const cv::Mat z = sampleTarget(frame, impl_->state[0], impl_->state[1], impl_->state[2],
                                 impl_->state[3], kTemplateFactor, kTemplateSize, resize_factor);
  impl_->template_ = preprocess(z);
  impl_->online_template_ = impl_->template_;
  impl_->online_max_template_ = impl_->template_;
  impl_->max_pred_score = -1.0f;
  impl_->frame_id = 0;
  template_updates_ = 0;
  return true;
}

DeepTrackResult DeepTracker::track(const cv::Mat& frame) {
  DeepTrackResult result;
  if (!loaded() || frame.empty()) return result;

  const auto t0 = std::chrono::steady_clock::now();

  ++impl_->frame_id;

  double resize_factor = 0.0;
  const cv::Mat search_patch =
      sampleTarget(frame, impl_->state[0], impl_->state[1], impl_->state[2], impl_->state[3],
                   kSearchFactor, kSearchSize, resize_factor);
  const std::vector<float> search = preprocess(search_patch);

  const std::array<int64_t, 4> shape_t{1, 3, kTemplateSize, kTemplateSize};
  const std::array<int64_t, 4> shape_s{1, 3, kSearchSize, kSearchSize};

  const std::array<Ort::Value, 3> inputs{
      Ort::Value::CreateTensor<float>(impl_->mem_info, const_cast<float*>(impl_->template_.data()),
                                      impl_->template_.size(), shape_t.data(), shape_t.size()),
      Ort::Value::CreateTensor<float>(
          impl_->mem_info, const_cast<float*>(impl_->online_template_.data()),
          impl_->online_template_.size(), shape_t.data(), shape_t.size()),
      Ort::Value::CreateTensor<float>(impl_->mem_info, const_cast<float*>(search.data()),
                                      search.size(), shape_s.data(), shape_s.size())};
  const char* input_names[] = {"img_t", "img_ot", "img_search"};
  const char* output_names[] = {"pred_boxes", "pred_scores"};

  std::vector<Ort::Value> outputs;
  try {
    outputs = impl_->session->Run(Ort::RunOptions{nullptr}, input_names, inputs.data(),
                                  inputs.size(), output_names, 2);
  } catch (const Ort::Exception& e) {
    std::fprintf(stderr, "DeepTracker: inference failed: %s\n", e.what());
    return result;
  }

  const float* pred_boxes = outputs[0].GetTensorData<float>();
  const float* pred_scores = outputs[1].GetTensorData<float>();
  const float score = pred_scores[0];

  // predict -> search-crop px -> image px (reference `map_box_back`).
  const double px =
      static_cast<double>(pred_boxes[0]) * static_cast<double>(kSearchSize) / resize_factor;
  const double py =
      static_cast<double>(pred_boxes[1]) * static_cast<double>(kSearchSize) / resize_factor;
  const double pw =
      static_cast<double>(pred_boxes[2]) * static_cast<double>(kSearchSize) / resize_factor;
  const double ph =
      static_cast<double>(pred_boxes[3]) * static_cast<double>(kSearchSize) / resize_factor;
  const double half_side = 0.5 * static_cast<double>(kSearchSize) / resize_factor;
  const double cx_prev = impl_->state[0] + 0.5 * impl_->state[2];
  const double cy_prev = impl_->state[1] + 0.5 * impl_->state[3];
  const double cx_real = px + (cx_prev - half_side);
  const double cy_real = py + (cy_prev - half_side);
  const double x1 = cx_real - 0.5 * pw;
  const double y1 = cy_real - 0.5 * ph;

  const std::array<double, 4> clipped = clipBox(x1, y1, pw, ph, frame.rows, frame.cols);
  impl_->state = clipped;

  // Online template update (matches reference).
  impl_->max_pred_score *= static_cast<float>(kMaxScoreDecay);
  if (score > 0.5f && score > impl_->max_pred_score) {
    double rf = 0.0;
    const cv::Mat z = sampleTarget(frame, impl_->state[0], impl_->state[1], impl_->state[2],
                                   impl_->state[3], kTemplateFactor, kTemplateSize, rf);
    impl_->online_max_template_ = preprocess(z);
    impl_->max_pred_score = score;
  }
  if (impl_->frame_id % kUpdateInterval == 0) {
    impl_->online_template_ = impl_->online_max_template_;
    impl_->max_pred_score = -1.0f;
    impl_->online_max_template_ = impl_->template_;
    ++template_updates_;
  }

  const double ms =
      std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
  last_ms_ = ms;

  result.bbox = cv::Rect(cv::Point(static_cast<int>(std::lround(impl_->state[0])),
                                   static_cast<int>(std::lround(impl_->state[1]))),
                         cv::Size(static_cast<int>(std::lround(impl_->state[2])),
                                  static_cast<int>(std::lround(impl_->state[3]))));
  result.score = score;
  result.valid = true;
  return result;
}

void DeepTracker::reset() {
  impl_->state = {0, 0, 0, 0};
  impl_->frame_id = 0;
  impl_->template_.clear();
  impl_->online_template_.clear();
  impl_->online_max_template_.clear();
  impl_->max_pred_score = -1.0f;
  template_updates_ = 0;
}