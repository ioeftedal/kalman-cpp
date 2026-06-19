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

#include "detector.hpp"
#include "flow_tracker.hpp"
#include "kalman/core/config_parser.hpp"
#include "kalman/core/filter.hpp"
#include "kalman/models/pixel.hpp"

namespace {

using kalman::PixelPolicy;
using kalman::UnscentedKalmanFilter;
using Filter = UnscentedKalmanFilter<double, 4, 2, PixelPolicy>;
using Vec2 = Eigen::Matrix<double, 2, 1>;
using Mat2 = Eigen::Matrix<double, 2, 2>;

using Mat24 = Eigen::Matrix<double, 2, 4>;
using Mat42 = Eigen::Matrix<double, 4, 2>;

// ---------------------------------------------------------------------------
//  Linear KF update with arbitrary H and R (sequential update helper)
// ---------------------------------------------------------------------------

void linearUpdate(Filter& filter, const Vec2& z, const Mat24& H, const Mat2& R) {
  auto x = filter.getState();
  auto P = filter.getCovariance();

  Vec2 nu = z - H * x;
  Mat2 S = H * P * H.transpose() + R;
  Mat42 K = S.ldlt().solve((P * H.transpose()).transpose()).transpose();

  x = x + K * nu;
  P = P - K * S * K.transpose();
  P = 0.5 * (P + P.transpose());

  filter.setState(x);
  filter.setCovariance(P);
}

// ---------------------------------------------------------------------------
//  Search ROI from predicted state and covariance
// ---------------------------------------------------------------------------

cv::Rect computeSearchROI(const Filter::VectorN& state, const Filter::MatrixN& cov,
                          cv::Size frame_size) {
  double u = state(0);
  double v = state(1);
  double sigma_u = std::max(50.0, 3.0 * std::sqrt(cov(0, 0)));
  double sigma_v = std::max(50.0, 3.0 * std::sqrt(cov(1, 1)));

  int x = std::max(0, static_cast<int>(u - sigma_u));
  int y = std::max(0, static_cast<int>(v - sigma_v));
  int w = std::min(frame_size.width - x, static_cast<int>(2.0 * sigma_u));
  int h = std::min(frame_size.height - y, static_cast<int>(2.0 * sigma_v));

  return cv::Rect(x, y, w, h);
}

// ---------------------------------------------------------------------------
//  Configuration
// ---------------------------------------------------------------------------

struct Config {
  std::string video_path = "videos/helicopter3.mp4";
  std::string model_path = "videos/yolov8n.onnx";
  float conf_threshold = 0.5f;
  std::string detections_csv = "videos/helicopter3.csv";

  double qc_cv = 500.0;
  double sigma_uv = 5.0;
  double init_pos_sigma = 100.0;
  double init_vel_sigma = 50.0;

  double sigma_klv = 10.0;       // KLT velocity measurement noise std-dev (px/s)
  double sigma_d_base = 1.0;     // KLT displacement noise baseline (pixels)
  double occ_noise_rate = 50.0;  // occlusion noise inflation rate (px²/s)
  int klt_min_features = 4;
  double gate_threshold = 9.21;    // Mahalanobis gate: chi-squared 2-DOF @ 99%
  double conf_noise_scale = 10.0;  // YOLO confidence noise scaling factor

  std::string output_video = "output/video_track.mp4";
  bool show_display = true;
};

Config loadConfig(const std::string& path) {
  Config cfg;
  kalman::parseConfigFile(path, [&](const std::string& key, const std::string& val) {
    if (key == "video_path")
      cfg.video_path = val;
    else if (key == "model_path")
      cfg.model_path = val;
    else if (key == "conf_threshold")
      cfg.conf_threshold = std::stof(val);
    else if (key == "detections_csv")
      cfg.detections_csv = val;
    else if (key == "qc_cv")
      cfg.qc_cv = std::stod(val);
    else if (key == "sigma_uv")
      cfg.sigma_uv = std::stod(val);
    else if (key == "init_pos_sigma")
      cfg.init_pos_sigma = std::stod(val);
    else if (key == "init_vel_sigma")
      cfg.init_vel_sigma = std::stod(val);
    else if (key == "sigma_klv")
      cfg.sigma_klv = std::stod(val);
    else if (key == "sigma_d_base")
      cfg.sigma_d_base = std::stod(val);
    else if (key == "occ_noise_rate")
      cfg.occ_noise_rate = std::stod(val);
    else if (key == "klt_min_features")
      cfg.klt_min_features = std::stoi(val);
    else if (key == "gate_threshold")
      cfg.gate_threshold = std::stod(val);
    else if (key == "conf_noise_scale")
      cfg.conf_noise_scale = std::stod(val);
    else if (key == "output_video")
      cfg.output_video = val;
    else if (key == "show_display")
      cfg.show_display = (val == "true");
    else
      std::fprintf(stderr, "Warning: unknown config key '%s'\n", key.c_str());
  });

  std::printf("Config: %s\n", path.c_str());
  std::printf("  video_path       = %s\n", cfg.video_path.c_str());
  std::printf("  model_path       = %s\n", cfg.model_path.c_str());
  std::printf("  conf_threshold   = %.2f\n", cfg.conf_threshold);
  std::printf("  detections_csv   = %s\n", cfg.detections_csv.c_str());
  std::printf("  qc_cv            = %.4f\n", cfg.qc_cv);
  std::printf("  sigma_uv         = %.2f px\n", cfg.sigma_uv);
  std::printf("  sigma_klv        = %.2f px/s\n", cfg.sigma_klv);
  std::printf("  sigma_d_base     = %.2f px\n", cfg.sigma_d_base);
  std::printf("  occ_noise_rate   = %.2f px^2/s\n", cfg.occ_noise_rate);
  std::printf("  klt_min_features = %d\n", cfg.klt_min_features);
  std::printf("  gate_threshold   = %.2f\n", cfg.gate_threshold);
  std::printf("  conf_noise_scale = %.1f\n", cfg.conf_noise_scale);
  std::printf("  init_pos_sigma   = %.1f px\n", cfg.init_pos_sigma);
  std::printf("  init_vel_sigma   = %.1f px/s\n", cfg.init_vel_sigma);
  std::printf("  output_video     = %s\n", cfg.output_video.c_str());
  std::printf("  show_display     = %s\n", cfg.show_display ? "yes" : "no");

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
};

void writeLog(const std::string& path, const std::vector<LogEntry>& entries) {
  FILE* csv = std::fopen(path.c_str(), "w");
  if (!csv) {
    std::fprintf(stderr, "Failed to open '%s'\n", path.c_str());
    return;
  }
  std::fprintf(csv,
               "frame,time,meas_u,meas_v,est_u,est_v,est_vu,est_vv,"
               "P_uu,P_vv,P_vuvu,P_vvvv,nees,"
               "klt_features,occ_counter,klt_disp_u,klt_disp_v\n");
  for (const auto& e : entries) {
    std::fprintf(csv,
                 "%d,%.3f,%.2f,%.2f,%.2f,%.2f,%.4f,%.4f,"
                 "%.8e,%.8e,%.8e,%.8e,%.8e,"
                 "%d,%d,%.2f,%.2f\n",
                 e.frame, e.time, e.meas_u, e.meas_v, e.est_u, e.est_v, e.est_vu, e.est_vv, e.P_uu,
                 e.P_vv, e.P_vuvu, e.P_vvvv, e.nees, e.klt_features, e.occ_counter, e.klt_disp_u,
                 e.klt_disp_v);
  }
  std::fclose(csv);
  std::printf("Wrote %s\n", path.c_str());
}

// ---------------------------------------------------------------------------
//  Drawing helpers
// ---------------------------------------------------------------------------

void drawHUD(cv::Mat& frame, int frame_idx, double time_s, const Filter::VectorN& state,
             bool has_detection, bool has_occlusion, int klt_features, int occ_counter) {
  char buf[256];
  std::snprintf(buf, sizeof(buf), "Frame: %d  Time: %.2f s", frame_idx, time_s);
  cv::putText(frame, buf, cv::Point(20, 40), cv::FONT_HERSHEY_SIMPLEX, 0.8,
              cv::Scalar(255, 255, 255), 2);

  std::snprintf(buf, sizeof(buf), "State: [%.0f, %.0f]  vel: [%.1f, %.1f] px/s", state(0), state(1),
                state(2), state(3));
  cv::putText(frame, buf, cv::Point(20, 70), cv::FONT_HERSHEY_SIMPLEX, 0.6,
              cv::Scalar(255, 255, 255), 1);

  if (has_occlusion) {
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
}

}  // namespace

// ===========================================================================

int main(int argc, char** argv) {
  std::string cfg_path = argc > 1 ? argv[1] : "config/video_track.cfg";
  const Config cfg = loadConfig(cfg_path);

  // -----------------------------------------------------------------------
  //  1. Open video
  // -----------------------------------------------------------------------
  cv::VideoCapture cap(cfg.video_path);
  if (!cap.isOpened()) {
    std::fprintf(stderr, "Failed to open video: %s\n", cfg.video_path.c_str());
    return 1;
  }

  double fps = cap.get(cv::CAP_PROP_FPS);
  if (fps <= 0) fps = 30.0;
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
    auto yolo = std::make_unique<YOLODetector>(cfg.model_path, cfg.conf_threshold);
    // Quick test: load a frame to verify detection works
    cv::Mat test_frame;
    cap >> test_frame;
    cap.set(cv::CAP_PROP_POS_FRAMES, 0);

    auto result = yolo->next(test_frame);
    if (result.has_value()) {
      std::printf("Detection: YOLO ONNX (live)\n");
      detector = std::move(yolo);
    } else {
      std::printf(
          "Detection: no target found in first frame with ONNX, "
          "trying CSV fallback...\n");
      auto csv = std::make_unique<CSVDetectionSource>(cfg.detections_csv);
      detector = std::move(csv);
    }
  }

  // -----------------------------------------------------------------------
  //  3. Initialise UKF
  // -----------------------------------------------------------------------
  Filter filter(0.1, 2.0, 0.0);
  filter.setProcessNoiseSpectralDensity(cfg.qc_cv);

  Mat2 R;
  R << cfg.sigma_uv * cfg.sigma_uv, 0.0, 0.0, cfg.sigma_uv * cfg.sigma_uv;
  filter.setMeasurementNoiseCovariance(R);

  bool initialized = false;
  int occ_counter = 0;

  // -----------------------------------------------------------------------
  //  3b. Initialise KLT flow tracker
  // -----------------------------------------------------------------------
  FlowTracker::Config flow_cfg;
  flow_cfg.min_features = cfg.klt_min_features;
  FlowTracker flow_tracker(flow_cfg);

  // Measurement projection matrices for sequential updates
  // H_yolo: extracts [u, v] from [u, v, vu, vv]
  Mat24 H_yolo;
  H_yolo.setZero();
  H_yolo(0, 0) = 1.0;
  H_yolo(1, 1) = 1.0;

  // H_klt: extracts [vu, vv] from [u, v, vu, vv]
  Mat24 H_klt;
  H_klt.setZero();
  H_klt(0, 2) = 1.0;
  H_klt(1, 3) = 1.0;

  // YOLO position measurement noise
  Mat2 R_yolo;
  R_yolo << cfg.sigma_uv * cfg.sigma_uv, 0.0, 0.0, cfg.sigma_uv * cfg.sigma_uv;
  filter.setMeasurementNoiseCovariance(R_yolo);

  // -----------------------------------------------------------------------
  //  4. Optional video writer
  // -----------------------------------------------------------------------
  cv::VideoWriter writer;
  bool save_video = !cfg.output_video.empty();
  if (save_video) {
    int fourcc = cv::VideoWriter::fourcc('m', 'p', '4', 'v');
    writer.open(cfg.output_video, fourcc, fps, cv::Size(width, height));
    if (!writer.isOpened()) {
      std::fprintf(stderr, "Warning: could not open VideoWriter\n");
      save_video = false;
    }
  }

  // -----------------------------------------------------------------------
  //  5. Main loop
  // -----------------------------------------------------------------------
  std::vector<LogEntry> log;
  log.reserve(total_frames);

  bool paused = false;
  int frame_idx = 0;

  while (true) {
    cv::Mat frame;
    cap >> frame;
    if (frame.empty()) break;

    double time_s = frame_idx / fps;

    // ---- KLT optical flow tracking (from frame_{t-1} to frame_t) ----
    FlowResult flow;
    if (initialized && flow_tracker.hasFeatures()) {
      // State is at predict(t|t-1) from previous frame's end-of-loop predict
      cv::Rect roi = computeSearchROI(filter.getState(), filter.getCovariance(), frame.size());
      flow = flow_tracker.track(frame, roi);
    }

    // ---- YOLO detection ----
    auto target = detector->next(frame);
    bool has_detection = target.has_value();

    // ---- UKF sequential updates ----
    if (initialized) {
      // (a) YOLO position update (when available)
      bool detection_accepted = false;
      cv::Point2f center(0, 0);
      double nees = 0.0;

      if (has_detection) {
        center = target->center();
        Vec2 z_y(center.x, center.y);
        float conf = target->confidence;

        // Confidence-adaptive measurement noise
        double conf_scale = 1.0 + cfg.conf_noise_scale * (1.0 - conf);
        Mat2 R_yolo_eff = R_yolo * conf_scale;

        // Innovation gating: reject outlier detections via Mahalanobis distance
        Vec2 innovation = z_y - H_yolo * filter.getState();
        Mat2 S_y = H_yolo * filter.getCovariance() * H_yolo.transpose() + R_yolo_eff;
        double mahalanobis_sq = innovation.transpose() * S_y.ldlt().solve(innovation);

        if (mahalanobis_sq < cfg.gate_threshold) {
          detection_accepted = true;

          nees = mahalanobis_sq;  // NEES = Mahalanobis distance squared
          linearUpdate(filter, z_y, H_yolo, R_yolo_eff);

          // (b) KLT velocity update (on top of YOLO position update)
          if (flow.valid) {
            Vec2 z_k(flow.mean_displacement.x / dt, flow.mean_displacement.y / dt);
            // Bidirectional KLT quality-adaptive sigma
            double sigma_v = cfg.sigma_klv;
            double uncertainty_ratio = (flow.uncertainty / dt) / cfg.sigma_klv;
            double quality_factor = std::min(1.0, flow.num_features / 15.0);
            if (uncertainty_ratio > 1.0) {
              sigma_v *= uncertainty_ratio;  // bad KLT: trust less
            } else {
              sigma_v *= (1.0 + 0.5 * (1.0 - quality_factor));  // good KLT: trust more
            }
            sigma_v = std::max(1.0, sigma_v);
            Mat2 R_klt;
            R_klt << sigma_v * sigma_v, 0.0, 0.0, sigma_v * sigma_v;
            linearUpdate(filter, z_k, H_klt, R_klt);
          }
        } else {
          has_detection = false;  // gated: treat as occlusion for HUD/logic
        }
      }

      if (detection_accepted) {
        log.push_back({frame_idx, time_s, center.x, center.y, filter.getState()(0),
                       filter.getState()(1), filter.getState()(2), filter.getState()(3),
                       filter.getCovariance()(0, 0), filter.getCovariance()(1, 1),
                       filter.getCovariance()(2, 2), filter.getCovariance()(3, 3), nees,
                       flow.num_features, occ_counter, flow.mean_displacement.x,
                       flow.mean_displacement.y});

        occ_counter = 0;

        // Refresh KLT features from current detection
        flow_tracker.refreshFeatures(frame, target->bbox);

        // Draw detection
        cv::circle(frame, center, 8, cv::Scalar(0, 255, 0), -1);
      } else {
        // No detection or rejected by gating: occlusion handling with KLT or predict-only
        if (flow.valid) {
          // KLT velocity update (cleaner than pseudo-position approach)
          Vec2 z_k(flow.mean_displacement.x / dt, flow.mean_displacement.y / dt);
          double occ_scale = 1.0 + cfg.occ_noise_rate * occ_counter / dt;
          double sigma_v = std::max(1.0, (flow.uncertainty / dt) * occ_scale);
          Mat2 R_klt;
          R_klt << sigma_v * sigma_v, 0.0, 0.0, sigma_v * sigma_v;
          linearUpdate(filter, z_k, H_klt, R_klt);
        }

        log.push_back({frame_idx, time_s, 0.0, 0.0, filter.getState()(0), filter.getState()(1),
                       filter.getState()(2), filter.getState()(3), filter.getCovariance()(0, 0),
                       filter.getCovariance()(1, 1), filter.getCovariance()(2, 2),
                       filter.getCovariance()(3, 3), 0.0, flow.num_features, occ_counter,
                       flow.mean_displacement.x, flow.mean_displacement.y});

        occ_counter++;
      }
    } else if (!initialized && has_detection) {
      // ---- First detection: initialise filter and KLT ----
      cv::Point2f center = target->center();
      Vec2 z(center.x, center.y);

      Filter::VectorN x_init;
      x_init << z(0), z(1), 0.0, 0.0;
      filter.setState(x_init);

      Filter::MatrixN P0 = Filter::MatrixN::Identity();
      double p_pos = cfg.init_pos_sigma * cfg.init_pos_sigma * 9.0;
      double p_vel = cfg.init_vel_sigma * cfg.init_vel_sigma * 9.0;
      P0(0, 0) = p_pos;
      P0(1, 1) = p_pos;
      P0(2, 2) = p_vel;
      P0(3, 3) = p_vel;
      filter.setCovariance(P0);

      // Initialise KLT features on the first detection
      flow_tracker.refreshFeatures(frame, target->bbox);

      initialized = true;
      std::printf("Initialised at frame %d: [%.1f, %.1f]\n", frame_idx, z(0), z(1));
    } else if (!initialized && frame_idx > 30) {
      std::fprintf(stderr, "No detection after 30 frames. Exiting.\n");
      break;
    }

    // ---- Predict for next frame (always after updates) ----
    if (initialized) {
      filter.predict(dt);
    }

    // ---- Draw prediction and KLT features ----
    if (initialized) {
      const auto& state = filter.getState();
      const auto& cov = filter.getCovariance();

      cv::Point2f pred(static_cast<int>(state(0)), static_cast<int>(state(1)));
      cv::circle(frame, pred, 10, cv::Scalar(0, 0, 255), 3);

      // Velocity arrow (scaled for visibility)
      cv::Point2f vel_tip(static_cast<int>(state(0) + state(2) * 10),
                          static_cast<int>(state(1) + state(3) * 10));
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
          if (err < 0.5f)
            color = cv::Scalar(0, 255, 0);  // green
          else if (err < 1.0f)
            color = cv::Scalar(0, 255, 255);  // yellow
          else
            color = cv::Scalar(0, 0, 255);  // red
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
    bool has_occlusion = (initialized && !has_detection && flow.valid);
    drawHUD(frame, frame_idx, time_s, initialized ? filter.getState() : Filter::VectorN::Zero(),
            has_detection, has_occlusion, flow.num_features, occ_counter);

    // ---- Display / write ----
    if (save_video) writer.write(frame);
    if (cfg.show_display) {
      cv::imshow("YOLO + UKF: Helicopter Tracking", frame);
      int key = cv::waitKey(1) & 0xFF;
      if (key == 'q') break;
      if (key == 'p') {
        paused = !paused;
        while (paused) {
          int k = cv::waitKey(30) & 0xFF;
          if (k == 'p' || k == 'q') {
            paused = false;
            if (k == 'q') break;
          }
        }
        if (key == 'q') break;
      }
    }

    ++frame_idx;
  }

  // -----------------------------------------------------------------------
  //  6. Cleanup
  // -----------------------------------------------------------------------
  cap.release();
  if (save_video) writer.release();
  if (cfg.show_display) cv::destroyAllWindows();

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
        total_sq_err += err_u * err_u + err_v * err_v;
        ++count;
        ++detection_count;
      }
      total_nees += e.nees;
      ++nees_count;
    }

    double rmse = std::sqrt(total_sq_err / std::max(count, 1));
    double mean_nees = total_nees / std::max(nees_count, 1);
    double track_pct = 100.0 * detection_count / std::max(nees_count, 1);

    int occlusion_frames = 0;
    int klt_active_frames = 0;
    for (const auto& e : log) {
      if (e.occ_counter > 0) occlusion_frames++;
      if (e.klt_features > 0) klt_active_frames++;
    }

    std::printf("\n=== Tracking Summary ===\n");
    std::printf("  Frames processed: %d\n", nees_count);
    std::printf("  Frames with detection: %d (%.1f%%)\n", detection_count, track_pct);
    std::printf("  Frames with KLT: %d (%.1f%%)\n", klt_active_frames,
                100.0 * klt_active_frames / std::max(nees_count, 1));
    std::printf("  Frames in occlusion: %d (%.1f%%)\n", occlusion_frames,
                100.0 * occlusion_frames / std::max(nees_count, 1));
    std::printf("  RMSE (vs detection): %.2f px\n", rmse);
    std::printf("  Mean NEES: %.2f (target: ~4.0 for 4-DOF)\n", mean_nees);

    std::string log_path = "output/video_track_log.csv";
    writeLog(log_path, log);

    std::printf("\nSaved: %s\n", log_path.c_str());
    if (save_video) std::printf("Saved: %s\n", cfg.output_video.c_str());
  }

  return 0;
}
