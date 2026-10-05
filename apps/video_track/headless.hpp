// Helpers for running the video_track apps on headless machines.
//
// Two failure modes this guards against:
//   1. OpenCV built without a GUI backend (e.g. Nix's `opencv` reports
//      "GUI: NONE") makes cv::imshow/cv::namedWindow throw cv::Exception,
//      which is unhandled and aborts the process mid-run.
//   2. Video writers and the CSV logs write into directories (e.g. `output/`)
//      that are not tracked in git, so the very first run of a fresh checkout
//      silently loses its output.
#pragma once

#include <cstdlib>
#include <filesystem>
#include <opencv2/core.hpp>
#include <string>

namespace video_track {

// True only when this OpenCV build has a GUI backend *and* a display server is
// reachable. Callers should still treat display errors as recoverable.
inline bool guiAvailable() {
  static const bool available = [] {
    const std::string info = cv::getBuildInformation();
    const std::size_t gui = info.find("GUI:");
    if (gui != std::string::npos) {
      const std::size_t eol = info.find('\n', gui);
      if (info.substr(gui, eol - gui).find("NONE") != std::string::npos) {
        return false;  // headless OpenCV build
      }
    }
#if defined(__linux__)
    return std::getenv("DISPLAY") != nullptr || std::getenv("WAYLAND_DISPLAY") != nullptr;
#else
    return true;
#endif
  }();
  return available;
}

// Creates the parent directory of `path` if it is missing. Returns false only
// when a directory could not be created (the caller decides how loud to be).
inline bool ensureParentDir(const std::string& path) {
  const std::filesystem::path parent = std::filesystem::path(path).parent_path();
  if (parent.empty()) {
    return true;
  }
  std::error_code ec;
  std::filesystem::create_directories(parent, ec);
  return !ec;
}

}  // namespace video_track