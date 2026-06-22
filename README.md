# kalman-cpp

**Unscented Kalman Filter (UKF) library for real-time tracking** — written in modern C++20 with Eigen3.

- **Header-only** (with optional pre-compiled instantiations)
- **Compile-time policy dispatch** selects process + measurement models with zero runtime overhead
- Models for **3-D bearing-only tracking** (pan/tilt from camera-centric and world-frame positions), **2-D pixel-space tracking** (CV and CA), and a **1-D thermal example**
- Ships with three applications: synthetic simulators and a **YOLO + KLT video tracker** using OpenCV
- Google Test suite with **NEES consistency validation**, and Google Benchmark suite

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

For pixel-space tracking (4-D CV, linear measurement model — uses closed-form KF update):

```cpp
#include <kalman/core/filter.hpp>
#include <kalman/models/pixel.hpp>

using PixelFilter = kalman::UnscentedKalmanFilter<double, 4, 2, kalman::PixelPolicy>;
PixelFilter filter(1.0, 2.0, 0.0);
```

---

## Project structure

```
├── include/kalman/
│   ├── core/                          # UKF engine
│   │   ├── fwd.hpp                    #   ProcessModel<T,N,Tag> forward decl
│   │   ├── concepts.hpp               #   KalmanPolicy concept (C++20)
│   │   ├── config_parser.hpp          #   Key=value config file parser
│   │   └── filter.hpp                 #   UnscentedKalmanFilter<T,N,M,Policy>
│   ├── kinematics/                    # Pure-math process models (no Policy deps)
│   │   ├── cv_3d.hpp                  #   6-D CV: computeF_6, computeQ_6
│   │   ├── ca_3d.hpp                  #   9-D CA: computeF_9, computeQ_9
│   │   ├── cv_pixel.hpp               #   4-D pixel CV: computeF_pixel, Q
│   │   └── ca_pixel.hpp               #   6-D pixel CA: computeF_pixel_CA, Q
│   └── models/                        # Policy structs (h, H, ModelTag, wrap)
│       ├── pan_tilt.hpp               #   PanTiltPolicy — non-linear bearing-only
│       ├── pixel.hpp                  #   PixelPolicy / PixelCAPolicy — linear 2-D
│       ├── world_frame.hpp            #   WorldFramePanTiltPolicy — extrinsics-aware
│       └── thermal.hpp                #   ThermalPolicy — 1-D CV example model
├── src/
│   ├── CMakeLists.txt                 # Library target (kalman)
│   └── unscented_kalman_filter.cpp    # Explicit template instantiations
├── apps/
│   ├── CMakeLists.txt                 # App build targets
│   ├── simulator/main.cpp             # 2-D linear KF, figure-8 trajectory
│   ├── drone_sim/main.cpp             # Pixel CV + CA UKF, Lissajous trajectory
│   └── video_track/                   # YOLOv8 + KLT optical flow + UKF
│       ├── CMakeLists.txt
│       ├── main.cpp                   #   Real video tracking pipeline
│       ├── detector.hpp / .cpp        #   YOLOv8 ONNX / CSV detection sources
│       └── flow_tracker.hpp / .cpp    #   KLT sparse optical flow tracker
├── config/                            # Default .cfg files for each app
│   ├── simulator.cfg
│   ├── drone_sim.cfg
│   └── video_track.cfg
├── tests/
│   ├── CMakeLists.txt                 # Test target: ukf_test
│   ├── helpers/
│   │   ├── model_test.hpp             #   Reusable test helpers (convergence, NEES)
│   │   └── trajectories.hpp           #   Figure-8 state generators
│   ├── models/
│   │   ├── pan_tilt_test.cpp          #   3-D CA + CV tests
│   │   ├── pixel_model_test.cpp       #   2-D pixel CV + CA tests
│   │   ├── world_frame_test.cpp       #   World-frame extrinsics tests
│   │   └── thermal_model_test.cpp     #   1-D thermal example tests
│   └── diag_test.cpp                  # Standalone diagnostic (not in test target)
├── benchmarks/
│   ├── CMakeLists.txt                 # Benchmark target: ukf_benchmark
│   └── ukf_benchmark.cpp              # Predict / update / filter throughput
├── scripts/                           # Python plotting scripts
│   ├── plot_common.py                 #   Shared matplotlib utilities
│   ├── plot_simulator.py              #   2-D KF simulator results
│   ├── plot_drone_sim.py              #   Pixel CV + CA filter results
│   ├── plot_video_track.py            #   Video tracking log
│   └── extract_detections.py          #   Extract YOLO detections to CSV
├── CMakeLists.txt                     # Root build — requires Eigen3, OpenCV, GTest, Benchmark
├── flake.nix                          # Nix flake for reproducible dev shell + build
├── pyproject.toml                     # Python dependencies for plotting scripts
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

| Model | N | State vector | PSD | Wiener process on |
|---|---|---|---|---|
| Constant-velocity (CV) | 6 | `[x, y, z, vx, vy, vz]ᵀ` | `q_c` | velocity |
| Constant-acceleration (CA) | 9 | `[x, y, z, vx, vy, vz, ax, ay, az]ᵀ` | `q_c` | jerk |

**Policies:** `PanTiltPolicy`, `WorldFramePanTiltPolicy`
- `PanTiltPolicy` — target state is already in the camera frame
- `WorldFramePanTiltPolicy` — target state is in a fixed world frame; set `setCameraPosition(x,y,z)` once and `setOrientation(pan, tilt)` before each `update()` call

### 2-D pixel-space tracking (u, v)

Measurement function (linear):

```
h(x) = [u, v]ᵀ      H = [I₂  0₂  …]
```

| Model | N | State vector | PSD | Wiener process on |
|---|---|---|---|---|
| CV | 4 | `[u, v, vu, vv]ᵀ` | `q_c` | velocity |
| CA | 6 | `[u, v, vu, vv, au, av]ᵀ` | `q_c` | jerk |

**Policies:** `PixelPolicy` (CV), `PixelCAPolicy` (CA)

### 1-D thermal tracking (example model)

| Model | N | State vector | PSD |
|---|---|---|---|
| CV | 2 | `[T, dT/dt]ᵀ` | `q_c` |

**Policy:** `ThermalPolicy` — included as minimal boilerplate example (see _Adding a model_ below)

---

## Architecture

The library has three layers:

```
┌─────────────────────────────────────────────────────┐
│  models/                  Policy structs            │
│  pan_tilt.hpp, pixel.hpp  (h, H, ModelTag, wrap)   │
├─────────────────────────────────────────────────────┤
│  kinematics/               Free functions           │
│  cv_3d.hpp, ca_3d.hpp     computeF(dt), computeQ() │
├─────────────────────────────────────────────────────┤
│  core/                     UKF engine               │
│  filter.hpp               UnscentedKalmanFilter     │
└─────────────────────────────────────────────────────┘
```

**How it works:**

1. `UnscentedKalmanFilter<T,N,M,Policy>` reads `Policy::ModelTag` to dispatch `ProcessModel<T,N,ModelTag>::F(dt)` and `Q(dt, qc)` at compile time
2. `Policy::isMeasurementLinear` selects at compile time between a closed-form KF update and the full UKF sigma-point update
3. `Policy::h(x)` defines the measurement model; `Policy::H(x)` provides the linearized observation matrix for linear models; `Policy::wrapInnovation(nu)` handles angular wrapping for bearing measurements

**Why the kinematics layer?** The `kinematics/` headers contain pure-math `constexpr` functions parameterized only on scalar type `T`. They are independent of the Policy / ModelTag machinery, making them reusable for other filter variants and unit-testable in isolation.

---

## Adding a new model

The `include/kalman/models/thermal.hpp` file is a complete working example. To add your own model:

1. **Define a ModelTag** (a unique type to disambiguate `ProcessModel` specializations):

   ```cpp
   struct MyTag {};
   ```

2. **Specialise `ProcessModel`** for your state dimension and tag:

   ```cpp
   template <typename T>
   struct ProcessModel<T, N, MyTag> {
     static constexpr auto F(T dt) noexcept { /* … */ }
     static constexpr auto Q(T dt, T q_c) noexcept { /* … */ }
   };
   ```

3. **Write a Policy struct**:

   ```cpp
   struct MyPolicy {
     using ModelTag = MyTag;
     static constexpr bool isMeasurementLinear = true;  // or false

     // For linear models:
     template <typename Derived>
     static constexpr auto H(const Eigen::MatrixBase<Derived>& x) noexcept { /* … */ }

     // For non-linear models (and always needed for tests):
     template <typename Derived>
     static constexpr auto h(const Eigen::MatrixBase<Derived>& x) noexcept { /* … */ }

     // Wrap angular innovations (no-op for most models):
     static void wrapInnovation(auto& nu) noexcept {}
   };
   ```

4. **Add an explicit instantiation** in `src/unscented_kalman_filter.cpp`:

   ```cpp
   template class UnscentedKalmanFilter<double, N, M, MyPolicy>;
   ```

That's it. The kinematics helpers are optional — you can inline `F` and `Q` directly in the `ProcessModel` specialisation (as `thermal.hpp` does).

---

## Building

### Dependencies

| Dependency | Version | Required for |
|---|---|---|
| C++23 compiler | (GCC 14+, Clang 18+, Apple Clang 16+) | all |
| Eigen3 | ≥ 3.4 | all |
| Google Test | ≥ 1.17 | tests (`ukf_test`) |
| Google Benchmark | ≥ 1.9 | benchmarks (`ukf_benchmark`) |
| OpenCV | ≥ 4 | `video_track` app |

### CMake

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build

# Run tests
ctest --test-dir build

# Run benchmarks
./build/benchmarks/ukf_benchmark
```

### Nix

```bash
nix develop                       # dev shell with all dependencies
nix build                         # build the library
```

### Optional: specify Eigen3 path

```bash
cmake -B build -DCMAKE_PREFIX_PATH=/path/to/eigen3
```

---

## Applications

| Executable | Build target | Description |
|---|---|---|
| `simulator` | `simulator` | 2-D linear KF tracking a figure-8 trajectory. Configurable via `config/simulator.cfg` |
| `drone_sim` | `drone_sim` | Pixel-space UKF (CV + CA model comparison) on a Lissajous trajectory. Includes asynchronous forward-prediction demo. Config: `config/drone_sim.cfg` |
| `video_track` | `video_track` | Real video tracking: YOLOv8 ONNX detections + KLT sparse optical flow fused by a 4-D pixel CV UKF. Config: `config/video_track.cfg` |
| `ukf_test` | `ukf_test` | Google Test suite — model structure, numerical stability, convergence, Monte Carlo NEES |
| `ukf_benchmark` | `ukf_benchmark` | Google Benchmark — predict/update/filter throughput at various dt values |

### Running the simulators

```bash
./build/apps/simulator              # uses config/simulator.cfg
./build/apps/drone_sim              # uses config/drone_sim.cfg
./build/apps/video_track            # uses config/video_track.cfg

# Custom config path as argv[1]
./build/apps/simulator my_config.cfg
```

Output CSVs are written to `output/` (configurable via `output_path` / `output_prefix` in the `.cfg` files).

### Plotting results

```bash
pip install -e .                    # installs matplotlib + numpy
python3 scripts/plot_simulator.py   # plots sim_output.csv
python3 scripts/plot_drone_sim.py   # plots drone_sim_output_cv.csv + _ca.csv
python3 scripts/plot_video_track.py # plots video_track_log.csv
```

---

## Testing

The library uses Google Test with reusable helper functions in `tests/helpers/`:

| Helper | What it checks |
|---|---|
| `testProcessModelStructure<Filter>(qc)` | `F(0) == I`, `Q` symmetric positive-semidefinite |
| `testNumericalStability<Filter>(qc)` | Works for `dt ∈ {1e-6, 1e6, -0.1, 0}` |
| `testConvergence<Filter>(states, qc, dt, σ, χ²_threshold)` | ANEES below chi-squared bound |
| `testNEESConsistency<Filter>(states, qc, dt, σ, χ²_lo, χ²_hi, n_runs)` | Monte Carlo NEES in [χ²_lo, χ²_hi] |

Run:

```bash
cmake -B build && cmake --build build && ctest --test-dir build --output-on-failure
```

To run a specific test:

```bash
./build/tests/ukf_test --gtest_filter="*NEES*"
```

---

## Benchmarking

```bash
./build/benchmarks/ukf_benchmark
```

Output is Google Benchmark JSON format. Throughput is measured for predict, update, and the combined filter step at multiple dt values (1 ms, 10 ms, 100 ms, 1 s) for both the 9-D 3-D pan/tilt model and the 4-D pixel-space model.

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
