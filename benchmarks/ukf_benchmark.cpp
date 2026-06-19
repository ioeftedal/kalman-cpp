/// @file
/// @brief Performance benchmarks for the Unscented Kalman Filter.
///
/// Uses the Google Benchmark library to measure throughput of the
/// predict, update, and combined filter steps for both the 9-D 3-D
/// pan/tilt model and the 4-D pixel-space CV model.
///
/// Build:  see CMakeLists.txt or build.ninja.
/// Run:    ./ukf_benchmark

#include <benchmark/benchmark.h>

#include <random>

#include "kalman/core/filter.hpp"
#include "kalman/models/pan_tilt.hpp"
#include "kalman/models/pixel.hpp"

namespace {

using Filter3D = kalman::UnscentedKalmanFilter<double, 9, 2, kalman::PanTiltPolicy>;
using FilterPixel = kalman::UnscentedKalmanFilter<double, 4, 2, kalman::PixelPolicy>;
using VectorM3 = Filter3D::VectorM;
using VectorM2 = FilterPixel::VectorM;

VectorM3 make_measurement_3d() {
  VectorM3 z;
  z << 0.1, 0.2;
  return z;
}

VectorM2 make_measurement_pixel() {
  VectorM2 z;
  z << 1920.0, 1080.0;
  return z;
}

}  // namespace

// -----------------------------------------------------------------------
//  3-D pan/tilt benchmarks
// -----------------------------------------------------------------------

/// Benchmark the UKF predict step for the 9-D 3-D model.
/// Parameterised by dt in milliseconds.
static void BM_Predict_3D(benchmark::State& state) {
  const double dt = state.range(0) / 1000.0;
  Filter3D filter;
  filter.setState(Filter3D::VectorN::Zero());
  filter.setCovariance(Filter3D::MatrixN::Identity());

  for (auto _ : state) {
    filter.predict(dt);
  }
}
BENCHMARK(BM_Predict_3D)->Arg(1)->Arg(10)->Arg(100)->Arg(1000)->Iterations(5000)->Repetitions(1);

/// Benchmark the UKF update step for the 9-D 3-D model.
static void BM_Update_3D(benchmark::State& state) {
  Filter3D filter;
  filter.setState(Filter3D::VectorN::Zero());
  filter.setCovariance(Filter3D::MatrixN::Identity());
  auto z = make_measurement_3d();

  for (auto _ : state) {
    filter.update(z);
  }
}
BENCHMARK(BM_Update_3D)->Iterations(5000)->Repetitions(1);

/// Benchmark the combined predict + update (filter) step for the 9-D 3-D model.
static void BM_Filter_3D(benchmark::State& state) {
  const double dt = 0.01;
  Filter3D filter;
  filter.setState(Filter3D::VectorN::Zero());
  filter.setCovariance(Filter3D::MatrixN::Identity());
  auto z = make_measurement_3d();

  for (auto _ : state) {
    filter.filter(dt, z);
  }
}
BENCHMARK(BM_Filter_3D)->Iterations(5000)->Repetitions(1);

// -----------------------------------------------------------------------
//  Pixel-space 4-D CV benchmarks
// -----------------------------------------------------------------------

/// Benchmark the UKF predict step for the 4-D pixel-space model.
static void BM_Predict_Pixel(benchmark::State& state) {
  const double dt = state.range(0) / 1000.0;
  FilterPixel filter;
  filter.setState(FilterPixel::VectorN::Zero());
  filter.setCovariance(FilterPixel::MatrixN::Identity());

  for (auto _ : state) {
    filter.predict(dt);
  }
}
BENCHMARK(BM_Predict_Pixel)->Arg(1)->Arg(10)->Arg(100)->Arg(1000)->Iterations(5000)->Repetitions(1);

/// Benchmark the UKF update step for the 4-D pixel-space model.
static void BM_Update_Pixel(benchmark::State& state) {
  FilterPixel filter;
  filter.setState(FilterPixel::VectorN::Zero());
  filter.setCovariance(FilterPixel::MatrixN::Identity());
  auto z = make_measurement_pixel();

  for (auto _ : state) {
    filter.update(z);
  }
}
BENCHMARK(BM_Update_Pixel)->Iterations(5000)->Repetitions(1);

/// Benchmark the combined predict + update for the 4-D pixel-space model.
static void BM_Filter_Pixel(benchmark::State& state) {
  const double dt = 0.01;
  FilterPixel filter;
  filter.setState(FilterPixel::VectorN::Zero());
  filter.setCovariance(FilterPixel::MatrixN::Identity());
  auto z = make_measurement_pixel();

  for (auto _ : state) {
    filter.filter(dt, z);
  }
}
BENCHMARK(BM_Filter_Pixel)->Iterations(5000)->Repetitions(1);

BENCHMARK_MAIN();
