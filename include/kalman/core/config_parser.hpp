#pragma once

#include <algorithm>
#include <cctype>
#include <cstdio>
#include <fstream>
#include <functional>
#include <sstream>
#include <string>

namespace kalman {

inline std::string trim(std::string s) {
  auto not_space = [](unsigned char c) { return !std::isspace(c); };
  auto front = std::find_if(s.begin(), s.end(), not_space);
  s.erase(s.begin(), front);
  auto back = std::find_if(s.rbegin(), s.rend(), not_space);
  s.erase(back.base(), s.end());
  return s;
}

inline bool parseConfigFile(
    const std::string& path,
    const std::function<void(const std::string&, const std::string&)>& handler) {
  std::ifstream file(path);
  if (!file.is_open()) {
    std::fprintf(stderr, "Warning: could not open '%s' — using defaults\n", path.c_str());
    return false;
  }

  std::string line;
  while (std::getline(file, line)) {
    line = trim(line);
    if (line.empty() || line[0] == '#') continue;

    auto eq = line.find('=');
    if (eq == std::string::npos) continue;

    std::string key = trim(line.substr(0, eq));
    std::string val = trim(line.substr(eq + 1));
    handler(key, val);
  }
  return true;
}

}  // namespace kalman
