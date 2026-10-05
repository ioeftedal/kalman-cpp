// Interactive ground-truth annotation tool.
//
//   video_track_annotate <video> <start_frame> <end_frame> [--out <csv>]
//       [--step N] [--overlay <run_log.csv>] [--resume <csv>]
//
// Frame numbers are 1-based VideoCapture indices (same convention as the
// `video_track` run log). Left-click sets the target centre on the current
// frame, right-click clears it. The output CSV has `frame,x,y` columns (x,y
// are the box centre, matching the GT format used by `video_track_eval`).
//
// `--dump <dir>` writes PNG composites (full frame + zoomed crop around the
// tracker estimate) for every `--step` frame and exits without showing a
// window; the companion `positions.txt` lists `frame,est_u,est_v` so crops
// can be anchored on the estimate while hand-typing ground truth from the
// saved PNG frames.

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <map>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>
#include <sstream>
#include <string>
#include <vector>

#include "headless.hpp"

namespace {

struct Point {
  double x = 0.0, y = 0.0;
};

void loadOverlay(std::map<int, Point>& out, const std::string& path) {
  std::ifstream file(path);
  if (!file.is_open()) {
    std::fprintf(stderr, "warning: cannot open overlay CSV: %s\n", path.c_str());
    return;
  }
  std::string line;
  std::getline(file, line);  // header
  std::map<std::string, int> col_index;
  {
    std::istringstream hs(line);
    std::string tok;
    int i = 0;
    while (std::getline(hs, tok, ',')) {
      col_index[tok] = i++;
    }
  }
  const int fi = (col_index.count("frame") != 0u) ? col_index["frame"] : -1;
  const int ui = (col_index.count("est_u") != 0u) ? col_index["est_u"] : -1;
  const int vi = (col_index.count("est_v") != 0u) ? col_index["est_v"] : -1;
  if (fi < 0 || ui < 0 || vi < 0) {
    std::fprintf(stderr, "error: overlay CSV needs frame,est_u,est_v columns\n");
    return;
  }
  while (std::getline(file, line)) {
    if (line.empty()) {
      continue;
    }
    std::istringstream ss(line);
    std::string tok;
    int col = 0;
    int idx = -1;
    float u = 0.0f;
    float v = 0.0f;
    while (std::getline(ss, tok, ',')) {
      if (col == fi) {
        idx = std::atoi(tok.c_str());
      }
      if (col == ui) {
        u = std::atof(tok.c_str());
      }
      if (col == vi) {
        v = std::atof(tok.c_str());
      }
      col++;
    }
    if (idx >= 0 && u > 0 && v > 0) {
      out[idx] = Point{static_cast<double>(u), static_cast<double>(v)};
    }
  }
}

void loadResume(std::map<int, Point>& pts, const std::string& path) {
  std::ifstream file(path);
  if (!file.is_open()) {
    std::fprintf(stderr, "error: cannot open resume CSV: %s\n", path.c_str());
    std::exit(1);
  }
  std::string line;
  std::getline(file, line);  // header
  while (std::getline(file, line)) {
    if (line.empty()) {
      continue;
    }
    std::istringstream ss(line);
    std::string tok;
    std::vector<double> cols;
    while (std::getline(ss, tok, ',')) {
      cols.push_back(std::atof(tok.c_str()));
    }
    if (cols.size() >= 3) {
      pts[static_cast<int>(cols[0])] = Point{cols[1], cols[2]};
    }
  }
  std::printf("resumed %zu existing points from %s\n", pts.size(), path.c_str());
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 4) {
    std::fprintf(stderr,
                 "usage: %s <video> <start_frame> <end_frame> [--out <csv>] "
                 "[--step N] [--overlay <run_log.csv>] [--resume <csv>]\n",
                 argv[0]);
    return 1;
  }

  const std::string video_path = argv[1];
  const int start_frame = std::atoi(argv[2]);
  const int end_frame = std::atoi(argv[3]);
  std::string out_path = "annotations.csv";
  int step = 1;
  std::string overlay_path;
  std::string resume_path;
  std::string dump_dir;

  for (int i = 4; i < argc; ++i) {
    if (std::strcmp(argv[i], "--out") == 0 && i + 1 < argc) {
      out_path = argv[++i];
    } else if (std::strcmp(argv[i], "--step") == 0 && i + 1 < argc) {
      step = std::atoi(argv[++i]);
    } else if (std::strcmp(argv[i], "--overlay") == 0 && i + 1 < argc) {
      overlay_path = argv[++i];
    } else if (std::strcmp(argv[i], "--resume") == 0 && i + 1 < argc) {
      resume_path = argv[++i];
    } else if (std::strcmp(argv[i], "--dump") == 0 && i + 1 < argc) {
      dump_dir = argv[++i];
    }
  }
  step = std::max(1, step);

  std::map<int, Point> pts;
  if (!resume_path.empty()) {
    loadResume(pts, resume_path);
  }

  std::map<int, Point> overlay;
  if (!overlay_path.empty()) {
    loadOverlay(overlay, overlay_path);
  }
  std::fprintf(stderr, "overlay entries: %zu\n", overlay.size());

  // Load the whole range into memory so navigation is instant (no seeks).
  cv::VideoCapture cap(video_path);
  if (!cap.isOpened()) {
    std::fprintf(stderr, "error: cannot open video: %s\n", video_path.c_str());
    return 1;
  }
  const int n_total = std::max(0, end_frame - start_frame + 1);
  std::vector<cv::Mat> frames;
  frames.reserve(n_total);
  cap.set(cv::CAP_PROP_POS_FRAMES, static_cast<double>(std::max(0, start_frame - 1)));
  cv::Mat raw;
  while (frames.size() < static_cast<size_t>(n_total) && cap.read(raw)) {
    frames.push_back(raw.clone());
  }
  if (frames.empty()) {
    std::fprintf(stderr, "error: cannot read frames %d..%d\n", start_frame, end_frame);
    return 1;
  }
  std::printf("loaded %zu frames (%d..%d)\n", frames.size(), start_frame,
              start_frame + static_cast<int>(frames.size()) - 1);
  const int last = static_cast<int>(frames.size()) - 1;

  if (!dump_dir.empty()) {
    const int crop = 384;     // crop side (full-res px) around the estimate
    const double zoom = 2.0;  // zoom factor for the crop panel
    const int full_w = 480;   // down-scaled full-frame panel width
    std::filesystem::create_directories(dump_dir);
    std::FILE* pos = std::fopen((dump_dir + "/positions.txt").c_str(), "w");
    if (pos == nullptr) {
      std::fprintf(stderr, "error: cannot write %s/positions.txt\n", dump_dir.c_str());
      return 1;
    }
    std::fprintf(pos, "frame,est_u,est_v,region_x,region_y\n");
    int written = 0;
    for (int i = 0; i <= last; i += step) {
      const int cur = start_frame + i;
      auto ov = overlay.find(cur);
      if (ov == overlay.end()) {
        continue;
      }
      const int cx = static_cast<int>(ov->second.x);
      const int cy = static_cast<int>(ov->second.y);

      cv::Mat full;
      cv::resize(frames[i], full, cv::Size(full_w, (full_w * frames[i].rows / frames[i].cols)), 0.0,
                 0.0, cv::INTER_AREA);
      double s = static_cast<double>(full.cols) / frames[i].cols;
      cv::drawMarker(full, cv::Point(static_cast<int>(cx * s), static_cast<int>(cy * s)),
                     cv::Scalar(0, 255, 0), cv::MARKER_CROSS, 24, 2);

      cv::Mat z;
      cv::Rect region(std::max(0, cx - (crop / 2)), std::max(0, cy - (crop / 2)), crop, crop);
      region.x = std::min(region.x, frames[i].cols - region.width);
      region.y = std::min(region.y, frames[i].rows - region.height);
      region.x = std::max(0, region.x);
      region.y = std::max(0, region.y);
      cv::resize(frames[i](region), z, cv::Size(0, 0), zoom, zoom, cv::INTER_LINEAR);

      cv::Mat view(std::max(full.rows, z.rows), full.cols + z.cols, full.type(),
                   cv::Scalar(20, 20, 20));
      full.copyTo(view(cv::Rect(0, 0, full.cols, full.rows)));
      z.copyTo(view(cv::Rect(full.cols, 0, z.cols, z.rows)));
      cv::putText(view, "full", cv::Point(8, 24), cv::FONT_HERSHEY_SIMPLEX, 0.7,
                  cv::Scalar(0, 255, 255), 2, cv::LINE_AA);
      cv::putText(view, "zoom x" + std::to_string(static_cast<int>(zoom)),
                  cv::Point(full.cols + 8, 24), cv::FONT_HERSHEY_SIMPLEX, 0.7,
                  cv::Scalar(0, 255, 255), 2, cv::LINE_AA);
      const std::string fname = dump_dir + "/f" + std::to_string(cur) + ".png";
      cv::imwrite(fname, view);
      std::fprintf(pos, "%d,%.2f,%.2f,%d,%d\n", cur, ov->second.x, ov->second.y, region.x,
                   region.y);
      written++;
    }
    std::fclose(pos);
    std::printf("dump: wrote %d composites %s/f*.png + positions.txt\n", written, dump_dir.c_str());
    return 0;
  }

  const std::string win = "annotate";
  if (!video_track::guiAvailable()) {
    std::fprintf(stderr,
                 "error: annotate needs an interactive display, but none is "
                 "available (headless OpenCV build or no DISPLAY/WAYLAND_DISPLAY)\n");
    return 1;
  }
  cv::namedWindow(win, cv::WINDOW_AUTOSIZE);

  struct Pt {
    std::map<int, Point>* points = nullptr;
    int* current = nullptr;
  } ptstate;
  ptstate.points = &pts;
  ptstate.current = nullptr;
  auto on_mouse = [](int event, int x, int y, int flags, void* userdata) {
    Pt* s = static_cast<Pt*>(userdata);
    if (event == cv::EVENT_LBUTTONDOWN) {
      (*s->points)[*s->current] = Point{static_cast<double>(x), static_cast<double>(y)};
      std::printf("f=%d  set    (%.1f, %.1f)\n", *s->current, static_cast<double>(x),
                  static_cast<double>(y));
    } else if (event == cv::EVENT_RBUTTONDOWN) {
      s->points->erase(*s->current);
      std::printf("f=%d  cleared\n", *s->current);
    }
    (void)flags;
  };
  cv::setMouseCallback(win, on_mouse, &ptstate);

  std::printf(
      "annotate %s frames %d..%d (step %d); left=set right=clear "
      "n/space/right=next p/left=prev s=save q=quit\n",
      video_path.c_str(), start_frame, end_frame, step);

  int pos = 0;  // index into `frames`
  bool done = false;
  while (!done) {
    const int cur = start_frame + pos;
    ptstate.current = const_cast<int*>(&cur);
    cv::Mat view = frames[pos].clone();

    if (pts.size() >= 2) {
      std::vector<cv::Point> poly;
      poly.reserve(pts.size());
      for (const auto& kv : pts) {
        poly.emplace_back(static_cast<int>(kv.second.x), static_cast<int>(kv.second.y));
      }
      for (size_t i = 1; i < poly.size(); ++i) {
        cv::line(view, poly[i - 1], poly[i], cv::Scalar(0, 255, 255), 1);
      }
      for (const auto& kv : pts) {
        cv::circle(view, cv::Point(static_cast<int>(kv.second.x), static_cast<int>(kv.second.y)), 3,
                   cv::Scalar(0, 255, 255), -1);
      }
    }

    auto ov = overlay.find(cur);
    if (ov != overlay.end()) {
      cv::drawMarker(view,
                     cv::Point(static_cast<int>(ov->second.x), static_cast<int>(ov->second.y)),
                     cv::Scalar(0, 255, 0), cv::MARKER_CROSS, 20, 2);
    }

    auto it = pts.find(cur);
    if (it != pts.end()) {
      cv::circle(view, cv::Point(static_cast<int>(it->second.x), static_cast<int>(it->second.y)), 6,
                 cv::Scalar(255, 0, 255), 2);
      cv::drawMarker(view,
                     cv::Point(static_cast<int>(it->second.x), static_cast<int>(it->second.y)),
                     cv::Scalar(255, 0, 255), cv::MARKER_CROSS, 30, 2);
    }

    char info[200];
    std::snprintf(info, sizeof(info),
                  "f %d/%d  %d%%  pts %zu   left=set right=clear n=next p=prev s=save q=quit", cur,
                  start_frame + last, static_cast<int>(100.0 * pos / (last + 1)), pts.size());
    cv::putText(view, info, cv::Point(10, 26), cv::FONT_HERSHEY_SIMPLEX, 0.7,
                cv::Scalar(255, 255, 255), 1, cv::LINE_AA);

    cv::imshow(win, view);

    const int key = cv::waitKey(0) & 0xff;
    if (key == 'q' || key == 27 || key == 'Q') {
      done = true;
    } else if (key == 'n' || key == ' ' || key == 65363) {  // n / space / right
      pos = std::min(last, pos + step);
    } else if (key == 'p' || key == 'b' || key == 65361) {  // p / b / left
      pos = std::max(0, pos - step);
    } else if (key == 's') {
      std::ofstream ofs(out_path, std::ios::trunc);
      if (!ofs.is_open()) {
        std::fprintf(stderr, "error: cannot write %s\n", out_path.c_str());
      } else {
        ofs << "frame,x,y\n";
        for (const auto& kv : pts) {
          ofs << kv.first << "," << std::round(kv.second.x) << "," << std::round(kv.second.y)
              << "\n";
        }
        std::printf("saved %zu points to %s\n", pts.size(), out_path.c_str());
      }
    }
  }
  cv::destroyWindow(win);

  std::ofstream ofs(out_path, std::ios::trunc);
  if (!ofs.is_open()) {
    std::fprintf(stderr, "error: cannot write %s\n", out_path.c_str());
    return 1;
  }
  ofs << "frame,x,y\n";
  std::vector<int> frames_out;
  frames_out.reserve(pts.size());
  for (const auto& kv : pts) {
    frames_out.push_back(kv.first);
  }
  std::sort(frames_out.begin(), frames_out.end());
  for (const int f : frames_out) {
    const Point& p = pts[f];
    ofs << f << "," << std::round(p.x) << "," << std::round(p.y) << "\n";
  }
  std::printf("wrote %zu points to %s\n", pts.size(), out_path.c_str());
  return 0;
}