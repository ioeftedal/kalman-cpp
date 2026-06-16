#include <Eigen/Dense>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <string>
#include <thread>
#include <vector>

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>

#include "detector.hpp"
#include "kalman/config_parser.hpp"
#include "kalman/pixel_measurement_model.hpp"
#include "kalman/unscented_kalman_filter.hpp"

namespace {

using kalman::PixelPolicy;
using kalman::UnscentedKalmanFilter;
using Filter = UnscentedKalmanFilter<double, 4, 2, PixelPolicy>;
using Vec2 = Eigen::Matrix<double, 2, 1>;
using Mat2 = Eigen::Matrix<double, 2, 2>;

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

  std::string output_video = "output/video_track.mp4";
  bool show_display = true;
};

Config loadConfig(const std::string& path) {
  Config cfg;
  kalman::parseConfigFile(path, [&](const std::string& key,
                                     const std::string& val) {
    if (key == "video_path")       cfg.video_path = val;
    else if (key == "model_path")  cfg.model_path = val;
    else if (key == "conf_threshold") cfg.conf_threshold = std::stof(val);
    else if (key == "detections_csv") cfg.detections_csv = val;
    else if (key == "qc_cv")       cfg.qc_cv = std::stod(val);
    else if (key == "sigma_uv")    cfg.sigma_uv = std::stod(val);
    else if (key == "init_pos_sigma") cfg.init_pos_sigma = std::stod(val);
    else if (key == "init_vel_sigma") cfg.init_vel_sigma = std::stod(val);
    else if (key == "output_video")   cfg.output_video = val;
    else if (key == "show_display")   cfg.show_display = (val == "true");
    else std::fprintf(stderr, "Warning: unknown config key '%s'\n",
                      key.c_str());
  });

  std::printf("Config: %s\n", path.c_str());
  std::printf("  video_path       = %s\n", cfg.video_path.c_str());
  std::printf("  model_path       = %s\n", cfg.model_path.c_str());
  std::printf("  conf_threshold   = %.2f\n", cfg.conf_threshold);
  std::printf("  detections_csv   = %s\n", cfg.detections_csv.c_str());
  std::printf("  qc_cv            = %.4f\n", cfg.qc_cv);
  std::printf("  sigma_uv         = %.2f px\n", cfg.sigma_uv);
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
};

void writeLog(const std::string& path, const std::vector<LogEntry>& entries) {
  FILE* csv = std::fopen(path.c_str(), "w");
  if (!csv) {
    std::fprintf(stderr, "Failed to open '%s'\n", path.c_str());
    return;
  }
  std::fprintf(csv,
      "frame,time,meas_u,meas_v,est_u,est_v,est_vu,est_vv,"
      "P_uu,P_vv,P_vuvu,P_vvvv,nees\n");
  for (const auto& e : entries) {
    std::fprintf(csv,
        "%d,%.3f,%.2f,%.2f,%.2f,%.2f,%.4f,%.4f,"
        "%.8e,%.8e,%.8e,%.8e,%.8e\n",
        e.frame, e.time,
        e.meas_u, e.meas_v,
        e.est_u, e.est_v, e.est_vu, e.est_vv,
        e.P_uu, e.P_vv, e.P_vuvu, e.P_vvvv, e.nees);
  }
  std::fclose(csv);
  std::printf("Wrote %s\n", path.c_str());
}

// ---------------------------------------------------------------------------
//  Drawing helpers
// ---------------------------------------------------------------------------

void drawHUD(cv::Mat& frame, int frame_idx, double time_s,
             const Filter::VectorN& state, bool has_detection) {
  char buf[256];
  std::snprintf(buf, sizeof(buf), "Frame: %d  Time: %.2f s", frame_idx, time_s);
  cv::putText(frame, buf, cv::Point(20, 40),
              cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(255, 255, 255), 2);

  std::snprintf(buf, sizeof(buf), "State: [%.0f, %.0f]  vel: [%.1f, %.1f] px/s",
                state(0), state(1), state(2), state(3));
  cv::putText(frame, buf, cv::Point(20, 70),
              cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(255, 255, 255), 1);

  std::snprintf(buf, sizeof(buf), "Detection: %s",
                has_detection ? "YES" : "NONE (predict-only)");
  cv::putText(frame, buf, cv::Point(20, 95),
              cv::FONT_HERSHEY_SIMPLEX, 0.6,
              has_detection ? cv::Scalar(0, 255, 0) : cv::Scalar(0, 0, 255), 1);
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

  std::printf("Video: %s  %dx%d  %.2f fps  %d frames\n",
              cfg.video_path.c_str(), width, height, fps, total_frames);

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
      std::printf("Detection: no target found in first frame with ONNX, "
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

    // ---- Detection ----
    auto target = detector->next(frame);

    // ---- UKF update + predict ----
    bool has_detection = target.has_value();
    if (has_detection) {
      Vec2 z(target->x, target->y);

      if (!initialized) {
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

        initialized = true;
        std::printf("Initialised at frame %d: [%.1f, %.1f]\n",
                    frame_idx, z(0), z(1));
      } else {
        // For NEES, compute innovation before update
        // The linear measurement model: H extracts [u, v]
        Eigen::Matrix<double, 2, 4> H;
        H.setZero();
        H(0, 0) = 1.0; H(1, 1) = 1.0;
        Vec2 z_pred = H * filter.getState();
        Vec2 innovation = z - z_pred;
        Mat2 S = H * filter.getCovariance() * H.transpose() + R;
        double nees = innovation.transpose() * S.ldlt().solve(innovation);

        filter.update(z);

        log.push_back({frame_idx, time_s, z(0), z(1),
                       filter.getState()(0), filter.getState()(1),
                       filter.getState()(2), filter.getState()(3),
                       filter.getCovariance()(0, 0),
                       filter.getCovariance()(1, 1),
                       filter.getCovariance()(2, 2),
                       filter.getCovariance()(3, 3), nees});
      }

      // Draw detection
      cv::circle(frame, cv::Point(static_cast<int>(z(0)),
                                   static_cast<int>(z(1))),
                 8, cv::Scalar(0, 255, 0), -1);
    }

    // ---- UKF predict (always, regardless of detection) ----
    if (initialized && !has_detection) {
      filter.predict(dt);

      const auto& state = filter.getState();
      const auto& cov = filter.getCovariance();
      log.push_back({frame_idx, time_s, 0.0, 0.0,
                     state(0), state(1), state(2), state(3),
                     cov(0, 0), cov(1, 1), cov(2, 2), cov(3, 3), 0.0});
    } else if (!initialized && frame_idx > 30) {
      std::fprintf(stderr, "No detection after 30 frames. Exiting.\n");
      break;
    }

    // ---- Predict for next frame (always after update) ----
    if (initialized && has_detection) {
      filter.predict(dt);
    }

    // ---- Draw prediction ----
    if (initialized) {
      const auto& state = filter.getState();
      const auto& cov = filter.getCovariance();

      cv::Point2f pred(static_cast<int>(state(0)),
                        static_cast<int>(state(1)));
      cv::circle(frame, pred, 10, cv::Scalar(0, 0, 255), 3);

      // Velocity arrow (scaled for visibility)
      cv::Point2f vel_tip(static_cast<int>(state(0) + state(2) * 10),
                           static_cast<int>(state(1) + state(3) * 10));
      cv::arrowedLine(frame, pred, vel_tip, cv::Scalar(255, 0, 0), 2);

      // Uncertainty ellipse (3-sigma)
      cv::ellipse(frame, pred,
                  cv::Size(static_cast<int>(3.0 * std::sqrt(cov(0, 0))),
                           static_cast<int>(3.0 * std::sqrt(cov(1, 1)))),
                  0.0, 0.0, 360.0,
                  cv::Scalar(0, 255, 255), 1, cv::LINE_AA);
    }

    // ---- HUD ----
    drawHUD(frame, frame_idx, time_s,
            initialized ? filter.getState() : Filter::VectorN::Zero(),
            has_detection);

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
          if (k == 'p' || k == 'q') { paused = false; if (k == 'q') break; }
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

    std::printf("\n=== Tracking Summary ===\n");
    std::printf("  Frames processed: %d\n", nees_count);
    std::printf("  Frames with detection: %d (%.1f%%)\n",
                detection_count, track_pct);
    std::printf("  RMSE (vs detection): %.2f px\n", rmse);
    std::printf("  Mean NEES: %.2f (target: ~4.0 for 4-DOF)\n", mean_nees);

    std::string log_path = "output/video_track_log.csv";
    writeLog(log_path, log);

    std::printf("\nSaved: %s\n", log_path.c_str());
    if (save_video) std::printf("Saved: %s\n", cfg.output_video.c_str());
  }

  return 0;
}
