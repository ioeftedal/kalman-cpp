// Video-tracking evaluation harness.
//
// Scores a `video_track` run log against a ground-truth reference trajectory
// and reports standard single-object-tracking metrics:
//   * RMSE / mean / median centre error (px)
//   * Precision curve (fraction of frames within a centre-distance threshold)
//     with Precision@10/@20/@50 and area-under-curve (AUC, 0..50 px)
//   * Success curve (fraction of frames with IoU above threshold) with
//     Success@0.50/@0.75 and AUC (IoU 0..1)
//
// The tracker emits a centre only, so the success metric re-centres the GT box
// geometry on the estimate.
//
// Usage:
//   video_track_eval <run_log.csv> <gt.csv> [--curves <prefix>]
//
// GT CSV columns: `frame,x,y,w,h`, where (x,y) is the box centre (w/h optional;
// zeros are replaced by the GT median box side). The run log is the CSV written
// by `video_track`.

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <map>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

namespace {

struct Point {
  double x = 0.0, y = 0.0;
};

struct GtBox {
  double x = 0.0, y = 0.0, w = 0.0, h = 0.0;
  bool has_bbox = false;
};

double iou(const GtBox& a, const GtBox& b) {
  const double ax1 = a.x - a.w / 2.0, ay1 = a.y - a.h / 2.0;
  const double ax2 = a.x + a.w / 2.0, ay2 = a.y + a.h / 2.0;
  const double bx1 = b.x - b.w / 2.0, by1 = b.y - b.h / 2.0;
  const double bx2 = b.x + b.w / 2.0, by2 = b.y + b.h / 2.0;
  const double ix = std::max(0.0, std::min(ax2, bx2) - std::max(ax1, bx1));
  const double iy = std::max(0.0, std::min(ay2, by2) - std::max(ay1, by1));
  if (ix <= 0.0 || iy <= 0.0) return 0.0;
  const double inter = ix * iy;
  const double uni = a.w * a.h + b.w * b.h - inter;
  if (uni <= 0.0) return 0.0;
  return inter / uni;
}

std::map<int, GtBox> readGt(const std::string& path, double& median_side) {
  std::map<int, GtBox> gt;
  std::ifstream file(path);
  if (!file.is_open()) {
    std::fprintf(stderr, "Error: cannot open GT CSV: %s\n", path.c_str());
    std::exit(1);
  }
  std::string line;
  std::getline(file, line);  // header
  std::vector<double> sides;
  while (std::getline(file, line)) {
    if (line.empty()) continue;
    std::istringstream ss(line);
    std::string tok;
    std::vector<double> cols;
    while (std::getline(ss, tok, ',')) cols.push_back(std::stod(tok));
    if (cols.size() < 3) continue;
    GtBox b;
    b.x = cols[1];
    b.y = cols[2];
    if (cols.size() >= 5 && cols[3] > 0.0 && cols[4] > 0.0) {
      b.w = cols[3];
      b.h = cols[4];
      b.has_bbox = true;
      sides.push_back(std::max(b.w, b.h));
    }
    gt[static_cast<int>(cols[0])] = b;
  }
  std::sort(sides.begin(), sides.end());
  median_side = sides.empty() ? 20.0 : sides[sides.size() / 2];
  return gt;
}

std::map<int, Point> readEst(const std::string& path) {
  std::map<int, Point> est;
  std::ifstream file(path);
  if (!file.is_open()) {
    std::fprintf(stderr, "Error: cannot open run log CSV: %s\n", path.c_str());
    std::exit(1);
  }
  std::string line;
  std::getline(file, line);  // header

  std::map<std::string, int> col;
  {
    std::istringstream ss(line);
    std::string name;
    int idx = 0;
    while (std::getline(ss, name, ',')) col[name] = idx++;
  }
  auto need = [&](const char* n) {
    auto it = col.find(n);
    if (it == col.end()) {
      std::fprintf(stderr, "Error: run log CSV missing column '%s'\n", n);
      std::exit(1);
    }
    return it->second;
  };
  const int c_frame = need("frame");
  const int c_est_u = need("est_u");
  const int c_est_v = need("est_v");

  while (std::getline(file, line)) {
    if (line.empty()) continue;
    std::istringstream ss(line);
    std::string tok;
    std::vector<std::string> cols;
    while (std::getline(ss, tok, ',')) cols.push_back(tok);
    if (cols.size() <= static_cast<size_t>(c_est_v)) continue;
    const int frame = std::stoi(cols[c_frame]);
    est[frame] = {std::stod(cols[c_est_u]), std::stod(cols[c_est_v])};
  }
  return est;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 3) {
    std::fprintf(stderr, "Usage: %s <run_log.csv> <gt.csv> [--curves <prefix>]\n", argv[0]);
    return 1;
  }
  const std::string log_path = argv[1];
  const std::string gt_path = argv[2];
  std::string curve_prefix;
  int range_lo = -1, range_hi = -1;
  if (argc >= 5 && std::string(argv[3]) == "--curves") curve_prefix = argv[4];
  for (int i = 3; i < argc; ++i) {
    if (std::strcmp(argv[i], "--range") == 0 && i + 2 < argc) {
      range_lo = std::atoi(argv[i + 1]);
      range_hi = std::atoi(argv[i + 2]);
      i += 2;
    }
  }

  double median_side = 20.0;
  const std::map<int, GtBox> gt = readGt(gt_path, median_side);
  const std::map<int, Point> est = readEst(log_path);

  std::vector<double> errors;
  std::vector<double> ious;
  for (const auto& kv : est) {
    if (range_lo >= 0 && kv.first < range_lo) continue;
    if (range_hi >= 0 && kv.first > range_hi) continue;
    auto git = gt.find(kv.first);
    if (git == gt.end()) continue;
    const double dx = kv.second.x - git->second.x;
    const double dy = kv.second.y - git->second.y;
    errors.push_back(std::sqrt(dx * dx + dy * dy));

    GtBox pred = git->second;
    if (!pred.has_bbox) pred.w = pred.h = median_side;
    pred.x = kv.second.x;
    pred.y = kv.second.y;
    ious.push_back(iou(git->second, pred));
  }

  const int n = static_cast<int>(errors.size());
  if (n == 0) {
    std::fprintf(stderr, "Error: no frames matched between run log and GT.\n");
    return 1;
  }

  auto count_lt = [](const std::vector<double>& v, double t) {
    return static_cast<int>(std::lower_bound(v.begin(), v.end(), t) - v.begin());
  };
  auto count_ge = [](const std::vector<double>& v, double t) {
    return static_cast<int>(v.end() - std::lower_bound(v.begin(), v.end(), t));
  };

  std::vector<double> sorted_errors = errors;
  std::sort(sorted_errors.begin(), sorted_errors.end());
  std::vector<double> sorted_ious = ious;
  std::sort(sorted_ious.begin(), sorted_ious.end());

  double sq = 0.0;
  for (double e : errors) sq += e * e;
  const double rmse = std::sqrt(sq / n);
  double sum = 0.0;
  for (double e : errors) sum += e;

  std::printf("\n=== Eval: %s  vs  %s ===\n", log_path.c_str(), gt_path.c_str());
  std::printf("  Frames scored:  %d\n", n);
  std::printf("  RMSE:           %.2f px\n", rmse);
  std::printf("  Mean error:     %.2f px\n", sum / n);
  std::printf("  Median error:   %.2f px\n", sorted_errors[n / 2]);
  std::printf("  Precision@10:   %.1f%%\n", 100.0 * count_lt(sorted_errors, 10.0) / n);
  std::printf("  Precision@20:   %.1f%%\n", 100.0 * count_lt(sorted_errors, 20.0) / n);
  std::printf("  Precision@50:   %.1f%%\n", 100.0 * count_lt(sorted_errors, 50.0) / n);
  std::printf("  Success@0.50:   %.1f%%\n", 100.0 * count_ge(sorted_ious, 0.50) / n);
  std::printf("  Success@0.75:   %.1f%%\n", 100.0 * count_ge(sorted_ious, 0.75) / n);

  double prec_auc = 0.0;
  const int prec_steps = 100;  // 0..50 px @ 0.5 px
  for (int i = 0; i < prec_steps; ++i) {
    const double t = i * 0.5;
    prec_auc += 100.0 * count_lt(sorted_errors, t) / n;
  }
  prec_auc /= static_cast<double>(prec_steps);
  std::printf("  Precision AUC (0..50 px): %.1f\n", prec_auc);

  double succ_auc = 0.0;
  const int succ_steps = 20;  // IoU 0..1 @ 0.05
  for (int i = 0; i < succ_steps; ++i) {
    const double t = i * 0.05;
    succ_auc += 100.0 * count_ge(sorted_ious, t) / n;
  }
  succ_auc /= static_cast<double>(succ_steps);
  std::printf("  Success AUC (IoU 0..1):  %.1f\n", succ_auc);

  if (!curve_prefix.empty()) {
    const std::string prec_path = curve_prefix + "_precision.csv";
    const std::string succ_path = curve_prefix + "_success.csv";
    FILE* f = std::fopen(prec_path.c_str(), "w");
    if (f) {
      std::fprintf(f, "threshold_px,precision\n");
      for (int i = 0; i <= prec_steps; ++i) {
        const double t = i * 0.5;
        std::fprintf(f, "%.1f,%.3f\n", t, 100.0 * count_lt(sorted_errors, t) / n);
      }
      std::fclose(f);
      std::printf("  Wrote %s\n", prec_path.c_str());
    }
    f = std::fopen(succ_path.c_str(), "w");
    if (f) {
      std::fprintf(f, "iou,success\n");
      for (int i = 0; i <= succ_steps; ++i) {
        const double t = i * 0.05;
        std::fprintf(f, "%.2f,%.3f\n", t, 100.0 * count_ge(sorted_ious, t) / n);
      }
      std::fclose(f);
      std::printf("  Wrote %s\n", succ_path.c_str());
    }
  }

  return 0;
}