cmake_minimum_required(VERSION 3.25)
project(kalman LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

option(KALMAN_PROFILING
       "Enable profiling build (optimized + debug symbols + frame pointer)" OFF)
option(KALMAN_USE_TRACY
       "Enable Tracy profiler instrumentation (implies KALMAN_PROFILING)" OFF)
option(KALMAN_TRACY_ON_DEMAND
       "Only collect Tracy data while a profiler is connected" OFF)
set(KALMAN_TRACY_CALLSTACK
    0
    CACHE STRING "Default Tracy zone callstack capture depth (0 = disabled)")

# Profiling build: optimized code with debug symbols for perf/callgrind/Tracy.
# Must be set before add_subdirectory so all targets inherit the options.
if(KALMAN_PROFILING OR KALMAN_USE_TRACY)
  include(CheckCXXCompilerFlag)
  check_cxx_compiler_flag("-march=native" HAS_MARCH_NATIVE)
  if(HAS_MARCH_NATIVE)
    add_compile_options(-O2 -g -fno-omit-frame-pointer -march=native)
  else()
    add_compile_options(-O2 -g -fno-omit-frame-pointer)
  endif()
endif()

enable_testing()

find_package(Eigen3 3.4 REQUIRED NO_MODULE)
find_package(
  OpenCV 4 REQUIRED
  COMPONENTS core
             imgproc
             imgcodecs
             videoio
             video
             highgui
             dnn)

# Tracy profiler dependency. Prefer an installed Tracy (e.g. the pinned nixpkgs
# `tracy` in the dev shell); otherwise fetch the pinned release archive so the
# build is fully reproducible (no git / tag drift).
if(KALMAN_USE_TRACY)
  find_package(Tracy QUIET)
  if(NOT Tracy_FOUND)
    include(FetchContent)
    FetchContent_Declare(
      tracy
      URL https://github.com/wolfpld/tracy/archive/refs/tags/v0.13.1.tar.gz
      URL_HASH
        SHA256=d4efc50ebcb0bfcfdbba148995aeb75044c0d80f5d91223aebfaa8fa9e563d2b
      DOWNLOAD_EXTRACT_TIMESTAMP TRUE)
    FetchContent_MakeAvailable(tracy)
    # Tracy is not clean under the project's strict -Werror flags (inherited via
    # CXXFLAGS) on recent compilers; relax only for Tracy.
    if(TARGET TracyClient)
      target_compile_options(TracyClient PRIVATE -Wno-error)
      if(KALMAN_TRACY_ON_DEMAND)
        target_compile_definitions(TracyClient PRIVATE TRACY_ON_DEMAND)
      endif()
      if(KALMAN_TRACY_CALLSTACK GREATER 0)
        target_compile_definitions(
          TracyClient PRIVATE TRACY_CALLSTACK=${KALMAN_TRACY_CALLSTACK})
      endif()
    endif()
  endif()
endif()

add_subdirectory(src)

add_subdirectory(apps)

find_package(GTest 1.17 REQUIRED)
add_subdirectory(tests)

find_package(benchmark 1.9 REQUIRED)
add_subdirectory(benchmarks)
