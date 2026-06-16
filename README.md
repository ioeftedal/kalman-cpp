# kalman-cpp

Unscented Kalman Filter (UKF) library for real-time tracking, written in modern C++23 with Eigen3.

Provides both **3-D bearing-only tracking** (pan/tilt from camera-centric position) and **2-D pixel-space tracking** (constant-velocity and constant-acceleration models). The library is header-only (with optional pre-compiled instantiations) and uses compile-time policy dispatch to select the process and measurement model.

## Models

### 3-D bearing-only (pan/tilt)

| Model | State dimension `N` | State vector | Process noise |
|---|---|---|---|
| Constant-acceleration (CA) | 9 | `[x, y, z, vx, vy, vz, ax, ay, az]ᵀ` | Wiener-process jerk (PSD q_c) |
| Constant-velocity (CV) | 6 | `[x, y, z, vx, vy, vz]ᵀ` | Wiener-process velocity (PSD q_c) |

**Coordinate convention** (camera-centric):
- `+x` — camera right
- `+y` — camera up
- `+z` — camera forward (optical axis)

**Measurement:** `h(x) = [atan2(x, z),  atan2(y, sqrt(x² + z²))]ᵀ`

### 2-D pixel-space (u, v)

| Model | State dimension `N` | State vector | Process noise |
|---|---|---|---|
| Constant-velocity (CV) | 4 | `[u, v, vu, vv]ᵀ` | Wiener-process velocity |
| Constant-acceleration (CA) | 6 | `[u, v, vu, vv, au, av]ᵀ` | Wiener-process jerk |

**Measurement:** `h(x) = [u, v]ᵀ` (linear extraction)

### Process models

All process models are derived from continuous-time Wiener-process acceleration (or jerk) models with power-spectral density `q_c`. The discrete-time state-transition matrix `F(dt)` and process-noise covariance `Q(dt, q_c)` are computed analytically via exact discretisation and are selected at compile-time via the `Policy::ModelTag`.

- CA (N=9): `F = [I, dt·I, ½dt²·I; 0, I, dt·I; 0, 0, I]`, Q is 5th-order in dt
- CV (N=6): `F = [I, dt·I; 0, I]`, Q is 3rd-order in dt
- Pixel CV (N=4): same pattern, 2-D blocks
- Pixel CA (N=6): same pattern, 2-D blocks

## Quick start

```cpp
#include <kalman/unscented_kalman_filter.hpp>

// 9-D pan/tilt filter
kalman::UnscentedKalmanFilter<double, 9, 2> ukf(1e-3, 2.0, 0.0);
ukf.setProcessNoiseSpectralDensity(5.0);

// Measurement noise covariance (pan and tilt std-dev in radians)
Eigen::Matrix2d R;
R << 0.01, 0.0, 0.0, 0.01;
ukf.setMeasurementNoiseCovariance(R);

// Filter loop
for (int i = 0; i < N; ++i) {
  Eigen::Vector2d z = getMeasurement(i);
  ukf.filter(dt, z);           // predict + update
  auto x = ukf.getState();     // state estimate
  auto P = ukf.getCovariance(); // covariance
}
```

For pixel-space tracking (4-D CV):

```cpp
using PixelFilter = kalman::UnscentedKalmanFilter<double, 4, 2, kalman::PixelPolicy>;
PixelFilter filter(1.0, 2.0, 0.0);
```

## Project structure

```
├── include/kalman/
│   ├── unscented_kalman_filter.hpp      # UKF template class
│   ├── pan_tilt_measurement_model.hpp   # 3-D bearing-only models
│   ├── pixel_measurement_model.hpp      # 2-D pixel-space models
│   ├── process_model_fwd.hpp            # Process model forward declaration
│   └── config_parser.hpp                # Config file parser utility
├── src/
│   ├── CMakeLists.txt                   # Library build target
│   └── unscented_kalman_filter.cpp      # Explicit instantiations
├── apps/
│   ├── CMakeLists.txt                   # Application build targets
│   ├── simulator/
│   │   └── main.cpp                     # 2-D linear KF simulator
│   └── drone_sim/
│       └── main.cpp                     # Drone pixel-tracker simulator
├── tests/
│   ├── CMakeLists.txt                   # Test build target
│   ├── pan_tilt_test.cpp                # 3-D pan/tilt UKF tests
│   └── pixel_model_test.cpp             # 2-D pixel CV + CA tests
├── benchmarks/
│   ├── CMakeLists.txt                   # Benchmark build target
│   └── ukf_benchmark.cpp                # Google Benchmark suite
├── config/
│   ├── simulator.cfg                    # Simulator configuration
│   └── drone_sim.cfg                    # Drone simulator configuration
├── scripts/                             # Python plotting scripts
├── output/                              # Generated CSV data and plots
└── CMakeLists.txt                       # Root CMake build
```

## Building

**Requirements:** C++23 compiler, Eigen3, Google Test, Google Benchmark (optional).

### CMake
```bash
cmake -B build -DCMAKE_PREFIX_PATH=/path/to/eigen3
cmake --build build
```

## Running

| Executable | Description |
|---|---|
| `simulator` | 2-D linear KF with figure-8 trajectory (`config/simulator.cfg`) |
| `drone_sim` | UKF (CV + CA) on pixel Lissajous (`config/drone_sim.cfg`) |
| `ukf_test`  | Google Test suite |
| `ukf_benchmark` | Google Benchmark suite |

### Simulators

Both simulators accept an optional config path as argv[1]; see `config/` for the default `.cfg` files.

Output CSVs and plots are written to `output/` and can be plotted with:
```bash
python3 scripts/plot_simulator.py
python3 scripts/plot_drone_sim.py
```

## Documentation

Generate API docs with Doxygen:

```bash
doxygen Doxyfile
# → docs/html/index.html
```

## References

- Julier, S.J. and Uhlmann, J.K., "A new extension of the Kalman filter to nonlinear systems", *Proc. SPIE*, 1997.
- Van der Merwe, R. and Wan, E.A., "The square-root unscented Kalman filter for state and parameter-estimation", *ICASSP*, 2001.
