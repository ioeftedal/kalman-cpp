#include <Eigen/Dense>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>
#include <string>
#include <thread>
#include <vector>

#include "appearance.hpp"
#include "detector.hpp"
#include "flow_tracker.hpp"
#include "global_motion.hpp"
#include "headless.hpp"
#ifdef KALMAN_ENABLE_DEEP
#include "deep_tracker.hpp"
#endif
#include "kalman/core/config_parser.hpp"
#include "kalman/core/imm.hpp"
#include "kalman/core/profiling.hpp"
#include "kalman/models/pixel.hpp"

namespace {

using ImmCvModel = kalman::ImmPixelModel<double, void>;
using ImmCaModel = kalman::ImmPixelModel<double, kalman::PixelCATag>;
using ImmCtModel = kalman::ImmPixelModel<double, kalman::PixelCTTag>;
using Imm = kalman::ImmFilter<double, ImmCvModel, ImmCaModel, ImmCtModel>;
using Vec2 = Eigen::Matrix<double, 2, 1>;
using Mat2 = Eigen::Matrix<double, 2, 2>;

// ---------------------------------------------------------------------------
//  Search ROI from predicted state and covariance
// ---------------------------------------------------------------------------

cv::Rect computeSearchROI(const Imm::VectorC& state, const Imm::MatrixC& cov, cv::Size frame_size) {
  KALMAN_PROFILE_ZONE_NAMED_C("VideoTrack::ROI", kalman::profiling::ColorApp);
  double u = state(0);
  double v = state(1);
  double sigma_u = std::max(50.0, 3.0 * std::sqrt(cov(0, 0)));
  double sigma_v = std::max(50.0, 3.0 * std::sqrt(cov(1, 1)));

  int x = std::max(0, static_cast<int>(u - sigma_u));
  int y = std::max(0, static_cast<int>(v - sigma_v));
  int w = std::min(frame_size.width - x, static_cast<int>(2.0 * sigma_u));
  int h = std::min(frame_size.height - y, static_cast<int>(2.0 * sigma_v));

  return {x, y, w, h};
}

// ---------------------------------------------------------------------------
//  Configuration
// ---------------------------------------------------------------------------

struct Config {
  std::string video_path = "videos/helicopter3.mp4";
  std::string model_path = "videos/yolov8n.onnx";
  float conf_threshold = 0.5f;
  std::string detections_csv = "videos/helicopter3.csv";
  int detector_input_size = 640;
  bool detector_letterbox = true;

  double qc_cv = 500.0;
  double sigma_uv = 5.0;
  double init_pos_sigma = 100.0;
  double init_vel_sigma = 50.0;

  double sigma_klv = 10.0;       // KLT velocity measurement noise std-dev (px/s)
  double occ_noise_rate = 50.0;  // occlusion noise inflation rate (px²/s)
  int klt_min_features = 4;
  double gate_threshold = 9.21;    // Mahalanobis gate: chi-squared 2-DOF @ 99%
  double conf_noise_scale = 10.0;  // YOLO confidence noise scaling factor

  double imm_transition_p = 0.02;  // IMM mode-transition off-diagonal probability
  double imm_q_mul_cv = 1.0;       // per-mode process-noise multipliers
  double imm_q_mul_ca = 0.8;
  double imm_q_mul_ct = 1.2;
  bool global_motion = true;  // camera-motion compensated KLT velocity
  bool adapt_noise = true;    // online measurement-noise adaptation
  int adapt_window = 30;

  bool appearance = true;            // template NCC reacquisition
  double appearance_min_ncc = 0.78;  // min TM_CCOEFF_NORMED peak to accept
  double appearance_min_psr = 3.0;   // min peak-to-sidelobe ratio to accept
  double appearance_update_rate = 0.15;
  double sigma_app = 12.0;        // appearance position noise std-dev (px)
  double appearance_gate = 20.0;  // IMM Mahalanobis gate for reacquisitions

  std::string output_video = "output/video_track.mp4";
  bool show_display = true;

  // Deep (MixFormerV2-S / ONNX Runtime) tracking backend
  std::string tracking_backend = "classical";  // "classical" | "deep"
  std::string deep_model_path = "videos/mixformer_v2_sim.onnx";
  double deep_sigma = 12.0;     // deep position measurement noise std-dev (px)
  double deep_score_min = 0.5;  // reject deep measurements below this score
  double deep_gate_px = 100.0;  // hard px acceptance radius vs predicted state
  int deep_reinit_frames = 60;  // low-score frames before re-init from detector
  int deep_detect_every = 1;    // run the detector every N frames (re-init monitor)

  std::string input_source = "video";  // "video" | "device" (live camera)
  int device_id = 0;
};

Config loadConfig(const std::string& path) {
  Config cfg;
  kalman::parseConfigFile(path, [&](const std::string& key, const std::string& val) {
    if (key == "video_path") {
      cfg.video_path = val;
    } else if (key == "model_path") {
      cfg.model_path = val;
    } else if (key == "conf_threshold") {
      cfg.conf_threshold = std::stof(val);
    } else if (key == "detections_csv") {
      cfg.detections_csv = val;
    } else if (key == "detector_input_size") {
      cfg.detector_input_size = std::stoi(val);
    } else if (key == "detector_letterbox") {
      cfg.detector_letterbox = (val == "true");
    } else if (key == "qc_cv") {
      cfg.qc_cv = std::stod(val);
    } else if (key == "sigma_uv") {
      cfg.sigma_uv = std::stod(val);
    } else if (key == "init_pos_sigma") {
      cfg.init_pos_sigma = std::stod(val);
    } else if (key == "init_vel_sigma") {
      cfg.init_vel_sigma = std::stod(val);
    } else if (key == "sigma_klv") {
      cfg.sigma_klv = std::stod(val);
    } else if (key == "occ_noise_rate") {
      cfg.occ_noise_rate = std::stod(val);
    } else if (key == "klt_min_features") {
      cfg.klt_min_features = std::stoi(val);
    } else if (key == "gate_threshold") {
      cfg.gate_threshold = std::stod(val);
    } else if (key == "conf_noise_scale") {
      cfg.conf_noise_scale = std::stod(val);
    } else if (key == "imm_transition_p") {
      cfg.imm_transition_p = std::stod(val);
    } else if (key == "imm_q_mul_cv") {
      cfg.imm_q_mul_cv = std::stod(val);
    } else if (key == "imm_q_mul_ca") {
      cfg.imm_q_mul_ca = std::stod(val);
    } else if (key == "imm_q_mul_ct") {
      cfg.imm_q_mul_ct = std::stod(val);
    } else if (key == "global_motion") {
      cfg.global_motion = (val == "true");
    } else if (key == "adapt_noise") {
      cfg.adapt_noise = (val == "true");
    } else if (key == "adapt_window") {
      cfg.adapt_window = std::stoi(val);
    } else if (key == "appearance") {
      cfg.appearance = (val == "true");
    } else if (key == "appearance_min_ncc") {
      cfg.appearance_min_ncc = std::stod(val);
    } else if (key == "appearance_min_psr") {
      cfg.appearance_min_psr = std::stod(val);
    } else if (key == "appearance_update_rate") {
      cfg.appearance_update_rate = std::stod(val);
    } else if (key == "sigma_app") {
      cfg.sigma_app = std::stod(val);
    } else if (key == "appearance_gate") {
      cfg.appearance_gate = std::stod(val);
    } else if (key == "output_video") {
      cfg.output_video = val;
    } else if (key == "show_display") {
      cfg.show_display = (val == "true");
    } else if (key == "tracking_backend") {
      cfg.tracking_backend = val;
    } else if (key == "deep_model_path") {
      cfg.deep_model_path = val;
    } else if (key == "deep_sigma") {
      cfg.deep_sigma = std::stod(val);
    } else if (key == "deep_score_min") {
      cfg.deep_score_min = std::stod(val);
    } else if (key == "deep_gate_px") {
      cfg.deep_gate_px = std::stod(val);
    } else if (key == "deep_reinit_frames") {
      cfg.deep_reinit_frames = std::stoi(val);
    } else if (key == "deep_detect_every") {
      cfg.deep_detect_every = std::stoi(val);
    } else if (key == "input_source") {
      cfg.input_source = val;
    } else if (key == "device_id") {
      cfg.device_id = std::stoi(val);
    } else {
      std::fprintf(stderr, "Warning: unknown config key '%s'\n", key.c_str());
    }
  });

  std::printf("Config: %s\n", path.c_str());
  std::printf("  video_path       = %s\n", cfg.video_path.c_str());
  std::printf("  model_path       = %s\n", cfg.model_path.c_str());
  std::printf("  conf_threshold   = %.2f\n", cfg.conf_threshold);
  std::printf("  detections_csv   = %s\n", cfg.detections_csv.c_str());
  std::printf("  detector_input   = %d px (letterbox %s)\n", cfg.detector_input_size,
              cfg.detector_letterbox ? "on" : "off");
  std::printf("  qc_cv            = %.4f\n", cfg.qc_cv);
  std::printf("  sigma_uv         = %.2f px\n", cfg.sigma_uv);
  std::printf("  sigma_klv        = %.2f px/s\n", cfg.sigma_klv);
  std::printf("  occ_noise_rate   = %.2f px^2/s\n", cfg.occ_noise_rate);
  std::printf("  klt_min_features = %d\n", cfg.klt_min_features);
  std::printf("  gate_threshold   = %.2f\n", cfg.gate_threshold);
  std::printf("  conf_noise_scale = %.1f\n", cfg.conf_noise_scale);
  std::printf("  init_pos_sigma   = %.1f px\n", cfg.init_pos_sigma);
  std::printf("  init_vel_sigma   = %.1f px/s\n", cfg.init_vel_sigma);
  std::printf("  imm_transition_p = %.4f\n", cfg.imm_transition_p);
  std::printf("  imm_q_mul        = CV %.2f CA %.2f CT %.2f\n", cfg.imm_q_mul_cv, cfg.imm_q_mul_ca,
              cfg.imm_q_mul_ct);
  std::printf("  global_motion    = %s\n", cfg.global_motion ? "yes" : "no");
  std::printf("  adapt_noise      = %s (window %d)\n", cfg.adapt_noise ? "yes" : "no",
              cfg.adapt_window);
  std::printf("  appearance       = %s (ncc %.2f, psr %.1f, lr %.2f, sigma %.1f px, gate %.1f)\n",
              cfg.appearance ? "yes" : "no", cfg.appearance_min_ncc, cfg.appearance_min_psr,
              cfg.appearance_update_rate, cfg.sigma_app, cfg.appearance_gate);
  std::printf("  output_video     = %s\n", cfg.output_video.c_str());
  std::printf("  show_display     = %s\n", cfg.show_display ? "yes" : "no");
  std::printf("  tracking_backend = %s\n", cfg.tracking_backend.c_str());
  if (cfg.tracking_backend == "deep") {
    std::printf("  deep_model       = %s\n", cfg.deep_model_path.c_str());
    std::printf(
        "  deep_sigma       = %.1f px  score_min %.2f  px_gate %.0f  reinit %d  detect_every %d\n",
        cfg.deep_sigma, cfg.deep_score_min, cfg.deep_gate_px, cfg.deep_reinit_frames,
        cfg.deep_detect_every);
  }

  return cfg;
}

// ---------------------------------------------------------------------------
//  CSV log entry
// ---------------------------------------------------------------------------

struct LogEntry {
  int frame;
  double time;
  double meas_u, meas_v;
  double est_u, est_v, est_vu, est_vv;
  double P_uu, P_vv, P_vuvu, P_vvvv;
  double nees;
  int klt_features;
  int occ_counter;
  double klt_disp_u, klt_disp_v;
  int appearance;
};

void writeLog(const std::string& path, const std::vector<LogEntry>& entries) {
  video_track::ensureParentDir(path);
  FILE* csv = std::fopen(path.c_str(), "w");
  if (csv == nullptr) {
    std::fprintf(stderr, "Failed to open '%s'\n", path.c_str());
    return;
  }
  std::fprintf(csv,
               "frame,time,meas_u,meas_v,est_u,est_v,est_vu,est_vv,"
               "P_uu,P_vv,P_vuvu,P_vvvv,nees,"
               "klt_features,occ_counter,klt_disp_u,klt_disp_v,appearance\n");
  for (const auto& e : entries) {
    std::fprintf(csv,
                 "%d,%.3f,%.2f,%.2f,%.2f,%.2f,%.4f,%.4f,"
                 "%.8e,%.8e,%.8e,%.8e,%.8e,"
                 "%d,%d,%.2f,%.2f,%d\n",
                 e.frame, e.time, e.meas_u, e.meas_v, e.est_u, e.est_v, e.est_vu, e.est_vv, e.P_uu,
                 e.P_vv, e.P_vuvu, e.P_vvvv, e.nees, e.klt_features, e.occ_counter, e.klt_disp_u,
                 e.klt_disp_v, e.appearance);
  }
  std::fclose(csv);
  std::printf("Wrote %s\n", path.c_str());
}

#ifdef KALMAN_ENABLE_DEEP
struct DeepRawEntry {
  int frame;
  float bx, by, bw, bh;
  float score;
  int accepted;
};

void writeDeepLog(const std::string& path, const std::vector<DeepRawEntry>& entries) {
  video_track::ensureParentDir(path);
  FILE* csv = std::fopen(path.c_str(), "w");
  if (!csv) {
    std::fprintf(stderr, "Failed to open '%s'\n", path.c_str());
    return;
  }
  std::fprintf(csv, "frame,box_x,box_y,box_w,box_h,score,accepted\n");
  for (const auto& e : entries) {
    std::fprintf(csv, "%d,%.2f,%.2f,%.2f,%.2f,%.4f,%d\n", e.frame, e.bx, e.by, e.bw, e.bh, e.score,
                 e.accepted);
  }
  std::fclose(csv);
  std::printf("Wrote %s\n", path.c_str());
}
#endif

// ---------------------------------------------------------------------------
//  Drawing helpers
// ---------------------------------------------------------------------------

void drawHUD(cv::Mat& frame, int frame_idx, double time_s, const Imm::VectorC& state,
             const Imm::ModeVec& mode_probs, bool gm_active, bool has_detection,
             bool from_appearance, bool has_occlusion, int klt_features, int occ_counter,
             const char* backend) {
  char buf[256];
  std::snprintf(buf, sizeof(buf), "Tracker: %s  Frame: %d  Time: %.2f s", backend, frame_idx,
                time_s);
  cv::putText(frame, buf, cv::Point(20, 40), cv::FONT_HERSHEY_SIMPLEX, 0.8,
              cv::Scalar(255, 255, 255), 2);

  std::snprintf(buf, sizeof(buf), "State: [%.0f, %.0f]  vel: [%.1f, %.1f] px/s", state(0), state(1),
                state(2), state(3));
  cv::putText(frame, buf, cv::Point(20, 70), cv::FONT_HERSHEY_SIMPLEX, 0.6,
              cv::Scalar(255, 255, 255), 1);

  if (from_appearance) {
    std::snprintf(buf, sizeof(buf), "REACQUIRED via appearance  KLT: %d feats", klt_features);
    cv::putText(frame, buf, cv::Point(20, 95), cv::FONT_HERSHEY_SIMPLEX, 0.6,
                cv::Scalar(0, 255, 255), 1);
  } else if (has_occlusion) {
    std::snprintf(buf, sizeof(buf), "OCCLUSION t=%d  KLT: %d feats", occ_counter, klt_features);
    cv::putText(frame, buf, cv::Point(20, 95), cv::FONT_HERSHEY_SIMPLEX, 0.6,
                cv::Scalar(0, 165, 255), 1);
  } else if (has_detection) {
    std::snprintf(buf, sizeof(buf), "Detection: YES  KLT: %d feats", klt_features);
    cv::putText(frame, buf, cv::Point(20, 95), cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 255, 0),
                1);
  } else {
    std::snprintf(buf, sizeof(buf), "Detection: NONE (predict-only)");
    cv::putText(frame, buf, cv::Point(20, 95), cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 0, 255),
                1);
  }

  // IMM mode probabilities (CV, CA, CT) and camera-motion compensation state
  std::snprintf(buf, sizeof(buf), "IMM: CV %.2f  CA %.2f  CT %.2f   %s",
                static_cast<double>(mode_probs(0)), static_cast<double>(mode_probs(1)),
                static_cast<double>(mode_probs(2)), gm_active ? "GM-COMP" : "");
  cv::putText(frame, buf, cv::Point(20, 120), cv::FONT_HERSHEY_SIMPLEX, 0.6,
              cv::Scalar(255, 255, 0), 1);
}

}  // namespace

// ===========================================================================

int main(int argc, char** argv) {
  KALMAN_PROFILE_SET_THREAD_NAME("VideoTrackMain");
  std::string cfg_path = argc > 1 ? argv[1] : "config/video_track.cfg";
  const Config cfg = loadConfig(cfg_path);

  // -----------------------------------------------------------------------
  //  1. Open video (or live capture device)
  // -----------------------------------------------------------------------
  cv::VideoCapture cap;
  if (cfg.input_source == "device") {
    std::printf("Input: live camera device %d\n", cfg.device_id);
    cap.open(cfg.device_id, cv::CAP_ANY);
  } else {
    cap.open(cfg.video_path);
  }
  if (!cap.isOpened()) {
    std::fprintf(stderr, "Failed to open input: %s\n",
                 cfg.input_source == "device" ? std::to_string(cfg.device_id).c_str()
                                              : cfg.video_path.c_str());
    return 1;
  }

  double fps = cap.get(cv::CAP_PROP_FPS);
  if (fps <= 0) {
    fps = 30.0;
  }
  double dt = 1.0 / fps;
  int total_frames = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_COUNT));
  int width = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
  int height = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));

  std::printf("Video: %s  %dx%d  %.2f fps  %d frames\n", cfg.video_path.c_str(), width, height, fps,
              total_frames);

  // -----------------------------------------------------------------------
  //  2. Create detection source
  // -----------------------------------------------------------------------
  std::unique_ptr<DetectionSource> detector;
  // Try ONNX model first, fall back to CSV
  {
    auto yolo = std::make_unique<YOLODetector>(cfg.model_path, cfg.conf_threshold, 0.45f,
                                               cfg.detector_input_size, cfg.detector_letterbox);
    // Probe the opening frames: a target may take a while to appear, so a
    // single-frame check falls over too easily. Rewind the capture after.
    const int probe_frames = (total_frames > 0) ? std::min(150, total_frames) : 150;
    bool yolo_usable = false;
    for (int i = 0; i < probe_frames; ++i) {
      cv::Mat probe_frame;
      cap >> probe_frame;
      if (probe_frame.empty()) {
        break;
      }
      if (yolo->next(probe_frame).has_value()) {
        yolo_usable = true;
        break;
      }
    }
    cap.set(cv::CAP_PROP_POS_FRAMES, 0);

    if (yolo_usable) {
      std::printf("Detection: YOLO ONNX (live)\n");
      detector = std::move(yolo);
    } else {
      std::printf(
          "Detection: no target found in first %d frames with ONNX, "
          "trying CSV fallback...\n",
          probe_frames);
      auto csv = std::make_unique<CSVDetectionSource>(cfg.detections_csv);
      detector = std::move(csv);
    }
  }

  // -----------------------------------------------------------------------
  //  3. Initialise IMM filter (CV / CA / CT)
  // -----------------------------------------------------------------------
  Imm imm;
  imm.setProcessNoiseSpectralDensities(
      {cfg.qc_cv * cfg.imm_q_mul_cv, cfg.qc_cv * cfg.imm_q_mul_ca, cfg.qc_cv * cfg.imm_q_mul_ct});
  imm.setTransitionProbabilities(cfg.imm_transition_p);
  imm.setAdaptiveNoise(cfg.adapt_noise, cfg.adapt_window);

  bool initialized = false;
  int occ_counter = 0;

  // -----------------------------------------------------------------------
  //  3b. Initialise KLT flow tracker and camera-motion estimator
  // -----------------------------------------------------------------------
  FlowTracker::Config flow_cfg;
  flow_cfg.min_features = cfg.klt_min_features;
  FlowTracker flow_tracker(flow_cfg);

  GlobalMotionEstimator gm(GlobalMotionEstimator::Config{});

  // Turn-rate prior for the IMM's coordinated-turn model, estimated from the
  // rotation of the (motion-compensated) optical-flow velocity across frames.
  Vec2 prev_vel(0.0, 0.0);
  bool have_prev_vel = false;
  double turn_rate = 0.0;

  // YOLO position measurement noise
  Mat2 R_yolo;
  R_yolo << cfg.sigma_uv * cfg.sigma_uv, 0.0, 0.0, cfg.sigma_uv * cfg.sigma_uv;

  // Appearance memory (template NCC matcher) for occlusion reacquisition
  kalman::tracking::AppearanceConfig app_cfg;
  app_cfg.min_ncc = cfg.appearance_min_ncc;
  app_cfg.min_psr = cfg.appearance_min_psr;
  app_cfg.update_rate = cfg.appearance_update_rate;
  kalman::tracking::TemplateMatcher matcher(app_cfg);
  int appearance_reacquired = 0;
  int appearance_attempts = 0;
  double appearance_max_ncc = 0.0;
  int appearance_valid_matches = 0;
  int appearance_gate_rejected = 0;

#ifdef KALMAN_ENABLE_DEEP
  DeepTracker deep_tracker;
  int deep_low_frames = 0;
  double deep_total_ms = 0.0;
  int deep_frames = 0;
  std::vector<DeepRawEntry> deep_raw;
#endif

  // Deep backend selection (falls back to classical if unavailable).
  bool deep_backend = (cfg.tracking_backend == "deep");
#ifdef KALMAN_ENABLE_DEEP
  if (deep_backend) {
    if (!deep_tracker.load(cfg.deep_model_path)) {
      std::fprintf(stderr, "Warning: deep tracker unavailable, using classical backend\n");
      deep_backend = false;
    }
  }
#else
  if (deep_backend) {
    std::fprintf(stderr,
                 "Warning: built without deep backend (KALMAN_ENABLE_ONNXRUNTIME), "
                 "using classical backend\n");
    deep_backend = false;
  }
#endif

  // -----------------------------------------------------------------------
  //  4. Optional video writer
  // -----------------------------------------------------------------------
  cv::VideoWriter writer;
  bool save_video = !cfg.output_video.empty();
  if (save_video) {
    video_track::ensureParentDir(cfg.output_video);
    // Codecs in decreasing order of preference. avc1/H.264 comes first because
    // it is the only MP4 codec every mainstream browser plays; MPEG-4 Part 2
    // (mp4v) is rejected by Firefox/Chrome/Safari with "no video with supported
    // format and mime type found". mp4v stays as the fallback for FFmpeg builds
    // without an H.264 encoder, MJPG as the last resort. Without this loop a
    // headless box with a stripped FFmpeg build would end up with no video at
    // all.
    static const char* kCodecs[] = {"avc1", "mp4v", "MJPG"};
    // True when the resulting file plays in a browser (Firefox/Chrome/Safari).
    static const bool kPlayable[] = {true, false, false};
    static const char* kLabels[] = {"avc1 (H.264)", "mp4v (MPEG-4 Part 2)", "MJPG"};
    int chosen = -1;
    for (int i = 0; i < 3; ++i) {
      const char* codec = kCodecs[i];
      const int fourcc = cv::VideoWriter::fourcc(codec[0], codec[1], codec[2], codec[3]);
      writer.open(cfg.output_video, fourcc, fps, cv::Size(width, height));
      if (writer.isOpened()) {
        chosen = i;
        break;
      }
    }
    if (chosen >= 0) {
      std::printf("  video writer    = %s\n", kLabels[chosen]);
      if (!kPlayable[chosen]) {
        std::fprintf(stderr,
                     "Warning: %s was written with %s, which browsers cannot play "
                     "(\"no video with supported format and mime type found\"); "
                     "open it in VLC/mpv instead, or install an FFmpeg build with "
                     "an H.264 encoder\n",
                     cfg.output_video.c_str(), kLabels[chosen]);
      }
    } else {
      std::fprintf(stderr,
                   "Warning: could not open VideoWriter for %s "
                   "(tried avc1/mp4v/MJPG)\n",
                   cfg.output_video.c_str());
      save_video = false;
    }
  }

  // A headless build or an unreachable X/Wayland session must not abort the
  // run: downgrade display to off instead.
  bool show_display = cfg.show_display;
  if (show_display && !video_track::guiAvailable()) {
    std::fprintf(stderr,
                 "Warning: show_display requested but no GUI is available "
                 "(headless OpenCV build or no DISPLAY/WAYLAND_DISPLAY); "
                 "continuing without display\n");
    show_display = false;
  }

  // -----------------------------------------------------------------------
  //  5. Main loop
  // -----------------------------------------------------------------------
  std::vector<LogEntry> log;
  log.reserve(total_frames);

  bool paused = false;
  int frame_idx = 0;

  while (true) {
    KALMAN_PROFILE_FRAME();
    const auto frame_t0 = std::chrono::steady_clock::now();
    cv::Mat frame;
    cap >> frame;
    if (frame.empty()) {
      break;
    }

    double time_s = frame_idx / fps;

    // ---- Camera-motion estimate for KLT velocity compensation ----
    GlobalMotion gm_motion;
    if (cfg.global_motion) {
      cv::Rect roi;
      if (initialized) {
        roi = computeSearchROI(imm.getState(), imm.getCovariance(), frame.size());
      }
      gm_motion = gm.track(frame, roi);
    }
    const bool gm_active = gm_motion.valid;

    // ---- KLT optical flow tracking (from frame_{t-1} to frame_t) ----
    FlowResult flow;
    if (initialized && flow_tracker.hasFeatures()) {
      KALMAN_PROFILE_ZONE_NAMED_C("VideoTrack::KLT_track", kalman::profiling::ColorMotion);
      // State is at predict(t|t-1) from previous frame's end-of-loop predict
      cv::Rect roi = computeSearchROI(imm.getState(), imm.getCovariance(), frame.size());
      flow = flow_tracker.track(frame, roi);
    }

    // Compensated target velocity measurement (px/frame), motion-compensated
    const bool flow_comp = flow.valid && gm_active;
    auto compensatedFlow = [&](const FlowResult& f) {
      cv::Point2f d = f.mean_displacement;
      if (flow_comp) {
        d -= gm_motion.displacement;
      }
      return d;
    };

    // ---- YOLO detection (throttled in deep mode: only used to (re)initialise) ----
    std::optional<Detection> target;
    const bool run_detector =
        (!deep_backend) || cfg.deep_detect_every <= 0 || (frame_idx % cfg.deep_detect_every == 0);
    if (run_detector) {
      KALMAN_PROFILE_ZONE_NAMED_C("VideoTrack::YOLO_detect", kalman::profiling::ColorVision);
      target = detector->next(frame);
    }
    const bool yolo_detected = target.has_value();
    bool has_detection = yolo_detected;
    bool from_appearance = false;

    // ---- IMM sequential updates ----
    if (initialized) {
      // Shared position update: mixture-gated IMM update plus the KLT velocity
      // update on top when a target motion measurement is available.
      // KLT/global-motion velocity measurement on top of any position fix.
      auto updateVelocityFromFlow = [&]() {
        if (!flow.valid) {
          return;
        }
        const cv::Point2f d = compensatedFlow(flow);
        Vec2 z_k(d.x / dt, d.y / dt);
        // Bidirectional KLT quality-adaptive sigma
        double sigma_v = cfg.sigma_klv;
        double uncertainty_ratio = (flow.uncertainty / dt) / cfg.sigma_klv;
        double quality_factor = std::min(1.0, flow.num_features / 15.0);
        if (uncertainty_ratio > 1.0) {
          sigma_v *= uncertainty_ratio;  // bad KLT: trust less
        } else {
          sigma_v *= (1.0 + (0.5 * (1.0 - quality_factor)));  // good KLT: trust more
        }
        sigma_v = std::max(1.0, sigma_v);
        Mat2 R_klt;
        R_klt << sigma_v * sigma_v, 0.0, 0.0, sigma_v * sigma_v;
        imm.updateVelocity(z_k, R_klt);
      };
      auto applyPositionUpdate = [&](const Vec2& z, const Mat2& R, double gate) -> bool {
        if (!imm.update(z, R, gate)) {
          return false;
        }
        updateVelocityFromFlow();
        return true;
      };

      // (a) YOLO position update (when available)
      bool detection_accepted = false;
      from_appearance = false;
      cv::Point2f center(0, 0);
      cv::Rect ref_bbox;
      bool have_ref_bbox = false;
      double nees = 0.0;

      if (!deep_backend && has_detection) {
        center = target->center();
        Vec2 z_y(center.x, center.y);
        float conf = target->confidence;

        // Confidence-adaptive measurement noise
        Mat2 R_yolo_eff = R_yolo * (1.0 + (cfg.conf_noise_scale * (1.0 - conf)));

        // Mixture-gated IMM position update: rejects outliers via the
        // mode-weighted innovation covariance.
        KALMAN_PROFILE_ZONE_NAMED_C("VideoTrack::IMM_update", kalman::profiling::ColorCore);
        if (applyPositionUpdate(Vec2(z_y), R_yolo_eff, cfg.gate_threshold)) {
          detection_accepted = true;
          nees = imm.lastGateDistance();  // NEES = normalised innovation
          ref_bbox = target->bbox;
          have_ref_bbox = true;
        } else {
          has_detection = false;  // gated: treat as occlusion for HUD/logic
        }
      }

#ifdef KALMAN_ENABLE_DEEP
      // (a2) Deep tracker measurement (MixFormerV2-S): the primary position
      // fix in deep mode. The detector only (re)initialises the template.
      if (deep_backend) {
        KALMAN_PROFILE_ZONE_NAMED_C("VideoTrack::deep", kalman::profiling::ColorVision);
        const DeepTrackResult d = deep_tracker.track(frame);
        ++deep_frames;
        if (d.valid) deep_total_ms += deep_tracker.lastInferenceMs();

        // (i) Sustained rejection/low confidence: resync. Prefer a fresh
        // detector box; otherwise resync the IMM to the confident deep box
        // itself, since the transformer carries its own localisation.
        if (deep_low_frames >= cfg.deep_reinit_frames && d.valid) {
          cv::Rect ibox = d.bbox;
          bool have_ibox = (d.score >= 0.7);
          if (yolo_detected) {
            ibox = target->bbox;
            have_ibox = true;
          }
          if (have_ibox) {
            const cv::Point2f ic(static_cast<float>(ibox.x + 0.5 * ibox.width),
                                 static_cast<float>(ibox.y + 0.5 * ibox.height));
            Imm::VectorC xi;
            xi << ic.x, ic.y, 0.0, 0.0;
            Imm::MatrixC P0 = Imm::MatrixC::Identity();
            double pp = cfg.init_pos_sigma * cfg.init_pos_sigma * 9.0;
            double pv = cfg.init_vel_sigma * cfg.init_vel_sigma * 9.0;
            P0(0, 0) = pp;
            P0(1, 1) = pp;
            P0(2, 2) = pv;
            P0(3, 3) = pv;
            imm.initialize(xi, P0);
            if (yolo_detected) {
              deep_tracker.init(frame, target->bbox);  // fresh template
              std::printf("Deep: resynced template+IMM at frame %d\n", frame_idx);
            } else {
              std::printf("Deep: resynced IMM to deep box at frame %d (score %.2f)\n", frame_idx,
                          d.score);
            }
            matcher.reset(frame, ibox);
            flow_tracker.refreshFeatures(frame, ibox);
            deep_low_frames = 0;
          }
        }

        // (ii) Deep position measurement: hard px gate first (score inflation
        // must not silently widen the Mahalanobis gate into accepting far-away
        // false locks), then the standard mixture-gated IMM update.
        bool d_accepted = false;
        if (d.valid && d.score >= cfg.deep_score_min) {
          const Vec2 z_d(d.bbox.x + 0.5 * d.bbox.width, d.bbox.y + 0.5 * d.bbox.height);
          const auto& st = imm.getState();
          const double du = std::abs(z_d(0) - st(0));
          const double dv = std::abs(z_d(1) - st(1));
          if (du <= cfg.deep_gate_px && dv <= cfg.deep_gate_px) {
            double sig = cfg.deep_sigma * (1.0 + 0.5 * cfg.conf_noise_scale * (1.0 - d.score));
            Mat2 R_d;
            R_d << sig * sig, 0.0, 0.0, sig * sig;
            // Hard association: the deep localiser is the measurement of
            // record; the IMM keeps its dynamics (for clean coasting during
            // invalidity) but its position is set directly by the tracker.
            const auto& pre_cov = imm.getCovariance();
            imm.setPosition(z_d, R_d);
            updateVelocityFromFlow();
            d_accepted = true;
            detection_accepted = true;
            center = cv::Point2f(static_cast<float>(z_d(0)), static_cast<float>(z_d(1)));
            nees = du * du / (pre_cov(0, 0) + sig * sig) + dv * dv / (pre_cov(1, 1) + sig * sig);
            ref_bbox = d.bbox;
            have_ref_bbox = true;
            deep_low_frames = 0;
          } else {
            ++deep_low_frames;
          }
        } else {
          ++deep_low_frames;
        }
        if (!detection_accepted) has_detection = false;

        if (d.valid) {
          deep_raw.push_back({frame_idx, static_cast<float>(d.bbox.x), static_cast<float>(d.bbox.y),
                              static_cast<float>(d.bbox.width), static_cast<float>(d.bbox.height),
                              d.score, d_accepted ? 1 : 0});
        }
      }
#endif

      // (b) Appearance reacquisition: when the detector misses or is gated
      // out, re-localise the target with the NCC template matcher inside a
      // search ROI around the predicted state and feed it as a measurement.
      if (!detection_accepted && cfg.appearance && matcher.hasTemplate()) {
        KALMAN_PROFILE_ZONE_NAMED_C("VideoTrack::appearance", kalman::profiling::ColorMotion);
        appearance_attempts++;
        const auto& st = imm.getState();
        const auto& c = imm.getCovariance();
        double margin = std::max((3.0 * std::sqrt(c(0, 0))) + (0.5 * matcher.lastNominalSize()),
                                 1.5 * matcher.lastNominalSize());
        margin = std::clamp(margin, 50.0, 400.0);
        cv::Rect roi(static_cast<int>(std::round(st(0) - margin)),
                     static_cast<int>(std::round(st(1) - margin)),
                     static_cast<int>(std::round(2.0 * margin)),
                     static_cast<int>(std::round(2.0 * margin)));
        const kalman::tracking::AppearanceMatch m = matcher.track(frame, roi);
        appearance_max_ncc = std::max(appearance_max_ncc, m.ncc);
        if (m.valid) {
          appearance_valid_matches++;
          Mat2 R_app;
          R_app << cfg.sigma_app * cfg.sigma_app, 0.0, 0.0, cfg.sigma_app * cfg.sigma_app;
          if (applyPositionUpdate(Vec2(m.position.x, m.position.y), R_app, cfg.appearance_gate)) {
            detection_accepted = true;
            from_appearance = true;
            center = m.position;
            nees = imm.lastGateDistance();
            ref_bbox = m.bbox;
            have_ref_bbox = true;
            appearance_reacquired++;
          } else {
            appearance_gate_rejected++;
          }
        }
      }

      if (detection_accepted) {
        log.push_back({frame_idx, time_s, center.x, center.y, imm.getState()(0), imm.getState()(1),
                       imm.getState()(2), imm.getState()(3), imm.getCovariance()(0, 0),
                       imm.getCovariance()(1, 1), imm.getCovariance()(2, 2),
                       imm.getCovariance()(3, 3), nees, flow.num_features, occ_counter,
                       flow.mean_displacement.x, flow.mean_displacement.y,
                       from_appearance ? 1 : 0});

        occ_counter = 0;

        if (have_ref_bbox) {
          // Keep the appearance memory current and refresh KLT features from
          // the accepted location (detection or reacquisition).
          matcher.update(frame, ref_bbox, cfg.appearance_update_rate);
          flow_tracker.refreshFeatures(frame, ref_bbox);
        }

        if (from_appearance) {
          cv::circle(frame, center, 8, cv::Scalar(0, 255, 255), -1);
          cv::rectangle(frame, ref_bbox, cv::Scalar(0, 255, 255), 1);
        } else {
          cv::circle(frame, center, 8, cv::Scalar(0, 255, 0), -1);
        }
      } else {
        // No detection or rejected by gating: occlusion handling with KLT or predict-only
        if (flow.valid) {
          // KLT velocity update (cleaner than pseudo-position approach)
          const cv::Point2f d = compensatedFlow(flow);
          Vec2 z_k(d.x / dt, d.y / dt);
          double occ_scale = 1.0 + (cfg.occ_noise_rate * occ_counter / dt);
          double sigma_v = std::max(1.0, (flow.uncertainty / dt) * occ_scale);
          Mat2 R_klt;
          R_klt << sigma_v * sigma_v, 0.0, 0.0, sigma_v * sigma_v;
          imm.updateVelocity(z_k, R_klt);
        }

        log.push_back({frame_idx, time_s, 0.0, 0.0, imm.getState()(0), imm.getState()(1),
                       imm.getState()(2), imm.getState()(3), imm.getCovariance()(0, 0),
                       imm.getCovariance()(1, 1), imm.getCovariance()(2, 2),
                       imm.getCovariance()(3, 3), 0.0, flow.num_features, occ_counter,
                       flow.mean_displacement.x, flow.mean_displacement.y, 0});

        occ_counter++;
      }
    } else if (!initialized && has_detection) {
      // ---- First detection: initialise IMM and KLT ----
      cv::Point2f center = target->center();
      Vec2 z(center.x, center.y);

      Imm::VectorC x_init;
      x_init << z(0), z(1), 0.0, 0.0;

      Imm::MatrixC P0 = Imm::MatrixC::Identity();
      double p_pos = cfg.init_pos_sigma * cfg.init_pos_sigma * 9.0;
      double p_vel = cfg.init_vel_sigma * cfg.init_vel_sigma * 9.0;
      P0(0, 0) = p_pos;
      P0(1, 1) = p_pos;
      P0(2, 2) = p_vel;
      P0(3, 3) = p_vel;
      imm.initialize(x_init, P0);

      // Initialise KLT features on the first detection
      flow_tracker.refreshFeatures(frame, target->bbox);
      matcher.reset(frame, target->bbox);
#ifdef KALMAN_ENABLE_DEEP
      if (deep_backend) {
        deep_tracker.init(frame, target->bbox);
        std::printf("Deep: initialised from detector at frame %d\n", frame_idx);
      }
#endif

      initialized = true;
      std::printf("Initialised at frame %d: [%.1f, %.1f]\n", frame_idx, z(0), z(1));
    } else if (!initialized && frame_idx > std::max(60, total_frames / 3)) {
      std::fprintf(stderr, "No detection within the first third of the video. Exiting.\n");
      break;
    }

    // ---- Predict for next frame (always after updates) ----
    if (initialized) {
      // Instantaneous turn rate from the velocity-direction change of the
      // compensated flow (y-down image coords: sign matches the CT motion).
      if (flow_comp) {
        const Vec2 v_now(compensatedFlow(flow).x / dt, compensatedFlow(flow).y / dt);
        if (have_prev_vel) {
          const double vn = v_now.norm();
          const double vp = prev_vel.norm();
          if (vn > 1e-3 && vp > 1e-3) {
            const double cross = (prev_vel.x() * v_now.y()) - (prev_vel.y() * v_now.x());
            turn_rate = (0.9 * turn_rate) + (0.1 * (cross / (vp * vn) / dt));
          }
        }
        prev_vel = v_now;
        have_prev_vel = true;
      } else {
        turn_rate *= 0.9;  // decay toward 0 while velocity is unknown
      }
      imm.setTurnRatePrior(turn_rate);
      imm.predict(dt);
    }

    // ---- Draw prediction and KLT features ----
    if (initialized) {
      KALMAN_PROFILE_ZONE_NAMED_C("VideoTrack::draw", kalman::profiling::ColorApp);
      const auto& state = imm.getState();
      const auto& cov = imm.getCovariance();

      cv::Point2f pred(static_cast<int>(state(0)), static_cast<int>(state(1)));
      cv::circle(frame, pred, 10, cv::Scalar(0, 0, 255), 3);

      // Velocity arrow (scaled for visibility)
      cv::Point2f vel_tip(static_cast<int>(state(0) + (state(2) * 10)),
                          static_cast<int>(state(1) + (state(3) * 10)));
      cv::arrowedLine(frame, pred, vel_tip, cv::Scalar(255, 0, 0), 2);

      // Uncertainty ellipse (3-sigma)
      cv::ellipse(frame, pred,
                  cv::Size(static_cast<int>(3.0 * std::sqrt(cov(0, 0))),
                           static_cast<int>(3.0 * std::sqrt(cov(1, 1)))),
                  0.0, 0.0, 360.0, cv::Scalar(0, 255, 255), 1, cv::LINE_AA);

      // Draw KLT feature points colored by quality
      // Green: low FB error (< 0.5 px), Yellow: moderate (0.5-1.0), Red: poor (> 1.0)
      if (flow.valid && flow.per_feature_error.size() == flow.feature_points.size()) {
        for (size_t i = 0; i < flow.feature_points.size(); ++i) {
          cv::Scalar color;
          float err = flow.per_feature_error[i];
          if (err < 0.5f) {
            color = cv::Scalar(0, 255, 0);  // green
          } else if (err < 1.0f) {
            color = cv::Scalar(0, 255, 255);  // yellow
          } else {
            color = cv::Scalar(0, 0, 255);  // red
          }
          cv::circle(frame, flow.feature_points[i], 3, color, -1);
        }
      } else {
        const auto& features = flow_tracker.getFeatures();
        for (const auto& pt : features) {
          cv::circle(frame, pt, 3, cv::Scalar(0, 255, 255), -1);
        }
      }

      // Draw mean flow vector
      if (flow.valid) {
        cv::Point2f flow_end(pred.x + static_cast<int>(flow.mean_displacement.x * 5),
                             pred.y + static_cast<int>(flow.mean_displacement.y * 5));
        cv::arrowedLine(frame, pred, flow_end, cv::Scalar(255, 255, 0), 2, cv::LINE_AA, 0, 0.3);
      }
    }

    // ---- HUD ----
    {
      KALMAN_PROFILE_ZONE_NAMED_C("VideoTrack::HUD", kalman::profiling::ColorApp);
      bool has_occlusion = (initialized && !has_detection && flow.valid);
      Imm::ModeVec mode_probs = Imm::ModeVec::Constant(1.0 / Imm::numModels());
      if (initialized) {
        mode_probs = imm.modeProbabilities();
      }
      drawHUD(frame, frame_idx, time_s, initialized ? imm.getState() : Imm::VectorC::Zero(),
              mode_probs, gm_active, has_detection, from_appearance, has_occlusion,
              flow.num_features, occ_counter, deep_backend ? "MixFormerV2" : "classical");
    }

    // ---- Display / write ----
    if (save_video) {
      writer.write(frame);
    }
    if (show_display) {
      // The window backend can still fail at imshow time (stale DISPLAY, no
      // window manager, ...); drop to headless instead of propagating.
      try {
        cv::imshow("YOLO + UKF: Helicopter Tracking", frame);
        int key = cv::waitKey(1) & 0xFF;
        if (key == 'q') {
          break;
        }
        if (key == 'p') {
          paused = !paused;
          while (paused) {
            int k = cv::waitKey(30) & 0xFF;
            if (k == 'p' || k == 'q') {
              paused = false;
              if (k == 'q') {
                break;
              }
            }
          }
          if (key == 'q') {
            break;
          }
        }
      } catch (const cv::Exception& e) {
        std::fprintf(stderr, "Warning: display unavailable (%s); continuing headless\n", e.what());
        show_display = false;
      }
    }

    const double frame_ms =
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - frame_t0)
            .count();
    KALMAN_PROFILE_PLOT("frame ms", frame_ms);

    ++frame_idx;
  }

  // -----------------------------------------------------------------------
  //  6. Cleanup
  // -----------------------------------------------------------------------
  cap.release();
  if (save_video) {
    writer.release();
  }
  if (show_display) {
    cv::destroyAllWindows();
  }

  // -----------------------------------------------------------------------
  //  7. Summary & log
  // -----------------------------------------------------------------------
  if (initialized) {
    double total_sq_err = 0.0;
    int count = 0;
    double total_nees = 0.0;
    int nees_count = 0;
    int detection_count = 0;

    for (const auto& e : log) {
      if (e.meas_u != 0.0 || e.meas_v != 0.0) {
        double err_u = e.est_u - e.meas_u;
        double err_v = e.est_v - e.meas_v;
        total_sq_err += (err_u * err_u) + (err_v * err_v);
        ++count;
        ++detection_count;
      }
      total_nees += e.nees;
      ++nees_count;
    }

    double rmse = std::sqrt(total_sq_err / std::max(count, 1));
    double track_pct = 100.0 * detection_count / std::max(nees_count, 1);

    // NEES is only meaningful on measurement-bearing frames (occlusion frames
    // carry nees = 0 and would dilute the mean).
    double total_detect_nees = 0.0;
    int detect_nees_count = 0;
    for (const auto& e : log) {
      if (e.meas_u != 0.0 || e.meas_v != 0.0) {
        total_detect_nees += e.nees;
        ++detect_nees_count;
      }
    }
    double mean_detect_nees = total_detect_nees / std::max(detect_nees_count, 1);

    int occlusion_frames = 0;
    int klt_active_frames = 0;
    for (const auto& e : log) {
      if (e.occ_counter > 0) {
        occlusion_frames++;
      }
      if (e.klt_features > 0) {
        klt_active_frames++;
      }
    }

    std::printf("\n=== Tracking Summary ===\n");
    std::printf("  Frames processed: %d\n", nees_count);
    std::printf("  Frames with detection: %d (%.1f%%)\n", detection_count, track_pct);
    std::printf("  Frames with KLT: %d (%.1f%%)\n", klt_active_frames,
                100.0 * klt_active_frames / std::max(nees_count, 1));
    std::printf("  Frames in occlusion: %d (%.1f%%)\n", occlusion_frames,
                100.0 * occlusion_frames / std::max(nees_count, 1));
    std::printf("  Frames reacquired via appearance: %d\n", appearance_reacquired);
    std::printf("  Appearance attempts: %d  valid %d  gate-rejected %d  (best ncc %.3f)\n",
                appearance_attempts, appearance_valid_matches, appearance_gate_rejected,
                appearance_max_ncc);
#ifdef KALMAN_ENABLE_DEEP
    if (deep_backend) {
      std::printf("  Deep tracker: avg %.1f ms/frame (%d frames, %.0f fps), %d template updates\n",
                  deep_frames ? deep_total_ms / deep_frames : 0.0, deep_frames,
                  deep_frames ? 1000.0 * deep_frames / std::max(deep_total_ms, 1e-9) : 0.0,
                  deep_tracker.templateUpdates());
    }
#endif
    std::printf("  RMSE (vs detection): %.2f px\n", rmse);
    std::printf("  Mean NEES: %.2f (target: ~2.0 for 2-DOF innovation)\n", mean_detect_nees);
    const auto& mu = imm.modeProbabilities();
    std::printf("  Mode probs: CV %.2f  CA %.2f  CT %.2f\n", static_cast<double>(mu(0)),
                static_cast<double>(mu(1)), static_cast<double>(mu(2)));
    std::printf("  Adaptive noise scale: %.2f\n", static_cast<double>(imm.noiseScale()));

    std::string log_path = "output/video_track_log.csv";
    writeLog(log_path, log);
#ifdef KALMAN_ENABLE_DEEP
    if (deep_backend) writeDeepLog("output/video_track_deep_raw.csv", deep_raw);
#endif

    std::printf("\nSaved: %s\n", log_path.c_str());
    if (save_video) {
      std::printf("Saved: %s\n", cfg.output_video.c_str());
    }
  }

  return 0;
}
