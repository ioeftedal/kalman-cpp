cmake_minimum_required(VERSION 3.25)
project(kalman LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
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

# Tracy profiler dependency
if(KALMAN_USE_TRACY)
  include(FetchContent)
  FetchContent_Declare(
    tracy
    GIT_REPOSITORY https://github.com/wolfpld/tracy.git
    GIT_TAG v0.11.1
    GIT_SHALLOW TRUE)
  FetchContent_MakeAvailable(tracy)
  # Tracy's code is not clean under the project's strict -Werror flags
  # (inherited via CXXFLAGS) on recent compilers; relax only for Tracy.
  if(TARGET TracyClient)
    target_compile_options(TracyClient PRIVATE -Wno-error)
  endif()
endif()

add_subdirectory(src)

add_subdirectory(apps)

find_package(GTest 1.17 REQUIRED)
add_subdirectory(tests)

find_package(benchmark 1.9 REQUIRED)
add_subdirectory(benchmarks)
