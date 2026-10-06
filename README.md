# kalman-cpp

**Unscented Kalman Filter (UKF) and Interacting Multiple Model (IMM) library for real-time tracking** — modern C++17 with Eigen3.

- **Compile-time policy dispatch** selects process + measurement models with zero runtime overhead
- **Header-only core** (with optional pre-compiled template instantiations) plus an offline-verifiable **UKF engine**
- Models for **3-D bearing-only tracking** (pan/tilt, camera-centric and world-frame), **2-D pixel-space tracking** (CV, CA, CT), and an **IMM filter** over pixel CV/CA/CT modes
- **`video_track`** — a real-time OpenCV pipeline: YOLOv8 ONNX detections + KLT optical flow + camera-motion compensation + template-matching reacquisition, fused by the IMM
- **Google Test suite** with ANEES consistency validation, and a Google Benchmark suite
- **Tracy profiler** instrumentation (pinned release, vendored via FetchContent)

---

## Quick start

```cpp
#include <kalman/core/filter.hpp>
#include <kalman/models/pan_tilt.hpp>

// 9-D constant-acceleration pan/tilt filter (N=9, M=2)
kalman::UnscentedKalmanFilter<double, 9, 2, kalman::PanTiltPolicy> ukf(1e-3, 2.0, 0.0);
ukf.setProcessNoiseSpectralDensity(5.0);

Eigen::Matrix2d R;
R << 0.01, 0.0, 0.0, 0.01;
ukf.setMeasurementNoiseCovariance(R);

// Filter loop
for (int i = 0; i < N; ++i) {
  Eigen::Vector2d z = getMeasurement(i);  // [pan, tilt] in radians
  ukf.filter(dt, z);                       // predict + update
  auto x = ukf.getState();                 // state estimate
  auto P = ukf.getCovariance();            // covariance
}
```

For pixel-space tracking (4-D CV, linear measurement model — closed-form KF update):

```cpp
#include <kalman/core/filter.hpp>
#include <kalman/models/pixel.hpp>

using PixelFilter = kalman::UnscentedKalmanFilter<double, 4, 2, kalman::PixelPolicy>;
PixelFilter filter(1.0, 2.0, 0.0);
```

### IMM fusion

```cpp
#include <kalman/core/imm.hpp>
#include <kalman/models/pixel.hpp>

using ImmCvModel = kalman::ImmPixelModel<double, void>;                 // CV
using ImmCaModel = kalman::ImmPixelModel<double, kalman::PixelCATag>;   // CA
using ImmCtModel = kalman::ImmPixelModel<double, kalman::PixelCTTag>;   // CT
using PixelIMM = kalman::ImmFilter<double, ImmCvModel, ImmCaModel, ImmCtModel>;

PixelIMM imm;
imm.setTransitionProbabilities(0.02);
imm.setProcessNoiseSpectralDensities({2000.0, 4000.0, 2500.0});
imm.setAdaptiveNoise(true, /* window= */ 30, /* target= */ 2.0);

// Filter loop: mix over the common 4-D state [u, v, vu, vv]ᵀ
for (...) {
  imm.predict(dt);                                  // mode mixing + predict
  if (imm.update(z, R, /* gate= */ 9.21)) { ... }   // gated measurement update
  auto x = imm.getState();                          // fused 4-D estimate
}
```

---

## Project structure

```
├── include/kalman/
│   ├── core/                          # UKF + IMM engine
│   │   ├── fwd.hpp                    #   ProcessModel<T,N,Tag> forward decl
│   │   ├── concepts.hpp               #   policy trait (C++17)
│   │   ├── constants.hpp              #   pi, physical/math constants
│   │   ├── config_parser.hpp          #   Key=value config file parser
│   │   ├── profiling.hpp              #   Tracy instrumentation macros
│   │   ├── filter.hpp                 #   UnscentedKalmanFilter<T,N,M,Policy>
│   │   └── imm.hpp                    #   ImmFilter — mode-mixing IMM over pixel modes
│   ├── kinematics/                    # Pure-math process models (no Policy deps)
│   │   ├── cv_3d.hpp / ca_3d.hpp      #   6-D CV / 9-D CA 3-D bearing-only
│   │   ├── cv_pixel.hpp / ca_pixel.hpp#   4-D CV / 6-D CA pixel-space
│   │   └── ct_pixel.hpp               #   5-D constant-turn pixel-space
│   └── models/                        # Policy structs (h, H, ModelTag, wrap)
│       ├── pan_tilt.hpp               #   PanTiltPolicy — non-linear bearing-only
│       ├── pixel.hpp                  #   Pixel{Policy,CAPolicy,CTPolicy}
│       └── world_frame.hpp            #   WorldFramePanTiltPolicy — extrinsics-aware
├── src/unscented_kalman_filter.cpp    # Explicit template instantiations
├── apps/video_track/
│   ├── main.cpp                       #   Real video tracking pipeline
│   ├── detector.hpp / .cpp            #   YOLOv8 ONNX / CSV detection sources
│   ├── flow_tracker.hpp / .cpp        #   KLT sparse optical flow tracker
│   ├── global_motion.hpp / .cpp       #   Camera-motion compensation
│   ├── appearance.hpp / .cpp          #   NCC template matching (reacquisition)
│   ├── deep_tracker.hpp / .cpp        #   MixFormer ONNX Runtime backend (optional)
│   ├── eval.cpp                       #   Offline tracking evaluation
│   └── annotate.cpp                   #   Frame annotation / video rendering
├── config/                            # video_track*.cfg (filter, IMM, KLT, ...)
├── tests/
│   ├── core/imm_test.cpp              #   IMM filtering + mode-mixing tests
│   ├── apps/                          #   global_motion, appearance tests
│   ├── models/                        #   pan_tilt, pixel, world_frame tests
│   └── helpers/                       #   Reusable convergence / ANEES helpers
├── benchmarks/ukf_benchmark.cpp       # Predict / update / filter throughput
├── videos/                            # Local input data (gitignored, see below)
├── CMakeLists.txt
├── flake.nix                          # Nix dev shell + shared tooling
├── .github/workflows/ci.yml           # GitHub Actions (build, test, lint)
└── Doxyfile                           # API documentation config
```

---

## Models

### 3-D bearing-only tracking (pan / tilt)

Camera-centric coordinate convention:

```
+x — camera right     +y — camera up     +z — camera forward (optical axis)
```

Measurement function (non-linear):

```
h(x) = [atan2(x, z),  atan2(y, sqrt(x² + z²))]ᵀ
```

| Model | N | State vector | Wiener process on |
|---|---|---|---|
| Constant-velocity (CV) | 6 | `[x, y, z, vx, vy, vz]ᵀ` | velocity |
| Constant-acceleration (CA) | 9 | `[x, y, z, vx, vy, vz, ax, ay, az]ᵀ` | jerk |

**Policies:** `PanTiltPolicy` (state in the camera frame), `WorldFramePanTiltPolicy` (state in a fixed world frame; set `setCameraPosition(x,y,z)` once and `setOrientation(pan, tilt)` before each `update()`).

### 2-D pixel-space tracking (u, v)

Linear measurement `h(x) = [u, v]ᵀ` for CV/CA; the CT mode measures heading angle as well.

| Model | N | State vector | Wiener process on |
|---|---|---|---|
| CV | 4 | `[u, v, vu, vv]ᵀ` | velocity |
| CA | 6 | `[u, v, vu, vv, au, av]ᵀ` | jerk |
| CT | 5 | `[u, v, vu, vv, ω]ᵀ` | velocity + turn rate |

The **IMM filter** (`include/kalman/core/imm.hpp`) mixes the three pixel modes over a common 4-D subspace `[u, v, vu, vv]ᵀ` with a mode-transition matrix, per-mode measurement-noise adaptive scaling, and innovation gating.

---

## Architecture

```
┌─────────────────────────────────────────────────────┐
│  models/                  Policy structs            │
│  pan_tilt.hpp, pixel.hpp  (h, H, ModelTag, wrap)   │
├─────────────────────────────────────────────────────┤
│  kinematics/               Free functions           │
│  cv_3d.hpp, ca_3d.hpp     computeF(dt), computeQ() │
├─────────────────────────────────────────────────────┤
│  core/                     UKF / IMM engine         │
│  filter.hpp, imm.hpp      UnscentedKalmanFilter     │
└─────────────────────────────────────────────────────┘
```

1. `UnscentedKalmanFilter<T,N,M,Policy>` reads `Policy::ModelTag` to dispatch `ProcessModel<T,N,Tag>::F(dt)` and `Q(dt, qc)` at compile time.
2. `Policy::isMeasurementLinear` selects between a closed-form KF update and the full sigma-point UKF update.
3. `Policy::h(x)` / `Policy::H(x)` define the measurement model; `Policy::wrapInnovation(nu)` handles angular wrapping.
4. `ImmFilter<T, Models...>` owns one sub-filter per mode; mixing weights follow the standard IMM recursion with per-mode innovation gating, adaptive measurement noise, and fused state/covariance output.

The `kinematics/` headers are pure-math `constexpr` functions parameterized only on scalar type `T`, independent of the Policy/ModelTag machinery, so they are reusable and testable in isolation.

---

## Adding a model

1. **Define a `ModelTag`** type to disambiguate `ProcessModel` specializations.
2. **Specialize `ProcessModel<T, N, MyTag>`** with `static constexpr F(T dt)` and `Q(T dt, T q_c)` (use/adapt the `kinematics/` helpers, or inline them — see `pixel.hpp` as the most self-contained example).
3. **Write a Policy struct** with `ModelTag`, `isMeasurementLinear`, `h(x)` (and `H(x)` when linear), and `wrapInnovation(nu)`.
4. **Add an explicit instantiation** in `src/unscented_kalman_filter.cpp`: `template class UnscentedKalmanFilter<double, N, M, MyPolicy>;`

---

## Building

### Dependencies

| Dependency | Version | Required for |
|---|---|---|
| C++17 compiler | GCC 9+, Clang 10+ | all |
| Eigen3 | ≥ 3.4 | all |
| Google Test | ≥ 1.17 | tests |
| Google Benchmark | ≥ 1.9 | benchmarks |
| OpenCV | ≥ 4 | `video_track` app + app tests |
| Tracy | pinned v0.13.1 (fetched) | optional profiling |
| ONNX Runtime | ≥ 1.27 | optional `deep_tracker` backend |

### CMake

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build

# Run tests
ctest --test-dir build --output-on-failure

# Run benchmarks
./build/benchmarks/ukf_benchmark
```

The default build enables Tracy instrumentation (`KALMAN_USE_TRACY=ON`); the pinned Tracy source is vendored via FetchContent, so no network happens during the build itself after configure. Disable with `-DKALMAN_USE_TRACY=OFF` for a lighter build. When kalman is consumed **as a subproject** (see below), all component defaults flip to OFF so the consumer is not forced to pull in OpenCV/GTest/benchmark/Tracy.

**Component options:**

| Option | Default | Effect |
|---|---|---|
| `KALMAN_BUILD_TESTS` | ON (top-level) / OFF (subproject) | Build GTest suite |
| `KALMAN_BUILD_APPS` | ON (top-level) / OFF (subproject) | Build `video_track` and tools |
| `KALMAN_BUILD_BENCHMARKS` | ON (top-level) / OFF (subproject) | Build benchmarks |
| `KALMAN_INSTALL` | ON (top-level) / OFF (subproject) | Generate install/export rules for the CMake package |
| `KALMAN_USE_TRACY` | ON (top-level) / OFF (subproject) | Tracy instrumentation (`-O2 -g -march=native`) |
| `KALMAN_TRACY_ON_DEMAND` | OFF | Only record while a profiler is connected |
| `KALMAN_ENABLE_ONNXRUNTIME` | OFF | Build `deep_tracker` backend (requires `ONNXRUNTIME_DIR`) |

### Using the library in another project

The core filtering code (`include/kalman/core`, `include/kalman/models`) has **no
dependency on OpenCV or video I/O** — the detector, frame capture, and rendering
all live in `apps/video_track/`. This makes the library easy to drop into a
project that brings its own detector (and runs live, from a camera stream rather
than a video file): your detector only has to emit (u, v) measurements plus a
per-measurement noise covariance, and the filter does the prediction.

**Option A — FetchContent / add_subdirectory**

```cmake
include(FetchContent)
FetchContent_Declare(kalman
  GIT_REPOSITORY https://github.com/ioeftedal/kalman-cpp
  GIT_TAG <release-or-commit>
  GIT_SHALLOW TRUE)
FetchContent_MakeAvailable(kalman)
target_link_libraries(your_app PRIVATE kalman::kalman)
```

Or vendor it as a git submodule and use `add_subdirectory(kalman-cpp)`. In both
cases only Eigen3 ≥ 3.4 is needed at configure time — apps, tests, benchmarks,
and Tracy all default to OFF for a subproject.

**Option B — find_package**

Build and install it once (Tracy must be off, so the export set stays clean):

```bash
cmake -S kalman-cpp -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DKALMAN_USE_TRACY=OFF \
  -DKALMAN_BUILD_APPS=OFF -DKALMAN_BUILD_TESTS=OFF -DKALMAN_BUILD_BENCHMARKS=OFF
cmake --install build --prefix /path/to/install/prefix
```

```cmake
find_package(kalman REQUIRED)
find_package(Eigen3 3.4 REQUIRED NO_MODULE)
target_link_libraries(your_app PRIVATE kalman::kalman)   # -DCMAKE_PREFIX_PATH=/path/to/install/prefix
```

`nix build` produces the equivalent package via the flake.

**Live prediction loop with your own detections**

```cpp
#include <kalman/core/imm.hpp>
#include <kalman/models/pixel.hpp>

using ImmCvModel = kalman::ImmPixelModel<double, void>;               // CV
using ImmCaModel = kalman::ImmPixelModel<double, kalman::PixelCATag>; // CA
using ImmCtModel = kalman::ImmPixelModel<double, kalman::PixelCTTag>; // CT
using PixelIMM = kalman::ImmFilter<double, ImmCvModel, ImmCaModel, ImmCtModel>;

PixelIMM imm;
imm.setTransitionProbabilities(0.02);
imm.setProcessNoiseSpectralDensities({2000.0, 4000.0, 2500.0});
imm.setAdaptiveNoise(true, /* window= */ 30, /* target= */ 2.0);

// Seed once, from your first detection (u, v): common state [u, v, vu, vv]
PixelIMM::VectorC x0;
x0 << u_px, v_px, 0.0, 0.0;
PixelIMM::MatrixC P0 = PixelIMM::MatrixC::Identity();
P0(0, 0) = P0(1, 1) = std::pow(10 * sigma_px, 2);   // position uncertainty
P0(2, 2) = P0(3, 3) = std::pow(200.0, 2);           // velocity uncertainty
imm.initialize(x0, P0);

auto last = std::chrono::steady_clock::now();
while (running) {
  auto det = your_detector.next(frame);   // your detector, live frames
  auto now = std::chrono::steady_clock::now();
  const double dt = std::chrono::duration<double>(now - last).count();
  last = now;

  imm.predict(dt);                        // mode mixing + time propagation

  if (det) {
    PixelIMM::VectorM z;
    z << det.center_u, det.center_v;
    PixelIMM::MatrixM R;
    R << sigma_px * sigma_px, 0.0, 0.0, sigma_px * sigma_px;
    if (imm.update(z, R, /* gate= */ 9.21)) {
      // accepted measurement
    }                                       // else: gated out -> coast
  }
  // No detection this frame? predict() still extrapolates the state.
  const auto x = imm.getState();          // fused [u, v, vu, vv]
  const auto P = imm.getCovariance();
}
```

For a single-model filter, swap the IMM for
`UnscentedKalmanFilter<double, 4, 2, kalman::PixelPolicy>` and call
`predict(dt)` / `update(z)` directly (closed-form KF update, linear model). The
`filter.hpp` and `imm.hpp` headers are header-only, so any scalar type (`double`,
`float`) and any policy from `models/` or your own custom model (see
"Adding a model") compile where they are used.

### Nix

```bash
nix develop          # dev shell with all dependencies + pre-commit hooks
nix build            # build + install the library (consumable via find_package)
```

Pre-commit hooks (clang-format, cmake-format, deadnix, nixfmt, typos) run automatically when committing inside the dev shell.

---

## Applications

| Executable | Build target | Description |
|---|---|---|
| `video_track` | `video_track` | Real-time video pipeline: YOLOv8 detections + KLT flow + motion compensation + NCC reacquisition + IMM |
| `video_track_eval` | `video_track_eval` | Offline evaluation of the tracking pipeline on `videos/` sequences |
| `video_track_annotate` | `video_track_annotate` | Render tracked boxes/trajectories back onto video |
| `ukf_test` | `ukf_test` (aka `ctest`) | Google Test suite — structure, stability, convergence, ANEES, IMM, global motion, appearance |
| `ukf_benchmark` | `ukf_benchmark` | Google Benchmark — predict/update/filter throughput at various dt |

### Running the video tracker

```bash
./build/apps/video_track            # uses config/video_track.cfg
```

The pipeline requires **local input data under `videos/`** (gitignored — listed in `config/video_track.cfg`):

- a source video, e.g. `videos/helicopter3.mp4`
- an ONNX YOLO model, e.g. `videos/yolov8n.onnx` (fallback: pre-extracted detections `videos/*.csv`)
- profiling config: `config/video_track_profile.cfg` (denser tracing)

The deep tracking backend (MixFormerV2-S) is optional: set `-DKALMAN_ENABLE_ONNXRUNTIME=ON -DONNXRUNTIME_DIR=/path/to/onnxruntime` and provide a model path in the config.

---

## Testing

The Google Test suite in `tests/` uses reusable helpers (`model_test.hpp`, `trajectories.hpp`):

| Helper | What it checks |
|---|---|
| `testProcessModelStructure<Filter>(qc)` | `F(0) == I`, `Q` symmetric positive-semidefinite |
| `testNumericalStability<Filter>(qc)` | Works for `dt ∈ {1e-6, 1e6, -0.1, 0}` |
| `testConvergence<Filter>(...)` | ANEES below the chi-squared bound |
| `testNEESConsistency<Filter>(...)` | Monte Carlo NEES within `[χ²_lo, χ²_hi]` |

Run:

```bash
cmake -B build && cmake --build build && ctest --test-dir build --output-on-failure
```

Run a specific case:

```bash
./build/tests/ukf_test --gtest_filter="*IMM*"
```

---

## Profiling with Tracy

Build with `-DKALMAN_USE_TRACY=ON` (default), then:

```bash
tracy-profiler          # from the nix dev shell
./build/apps/video_track config/video_track_profile.cfg
```

Tests and benchmarks are instrumented too. Use `KALMAN_TRACY_ON_DEMAND=ON` to avoid the capture overhead when no profiler is attached.

---

## Documentation

```bash
doxygen Doxyfile
open docs/html/index.html
```

---

## References

- Julier, S.J. and Uhlmann, J.K., "A new extension of the Kalman filter to nonlinear systems", _Proc. SPIE_, 1997.
- Van der Merwe, R. and Wan, E.A., "The square-root unscented Kalman filter for state and parameter-estimation", _ICASSP_, 2001.
- Li, X.R. and Jilkov, V.P., "Survey of maneuvering target tracking. Part V: Multiple-model methods", _IEEE Trans. AES_, 2005.