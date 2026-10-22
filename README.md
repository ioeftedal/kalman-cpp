# kalman-cpp

**Single-target pixel-space tracker for a fixed-rate pan/tilt camera** — the
prediction half of a live drone-tracking pipeline, meant to be linked as a
library by the code that runs your object detector.

- Camera: stationary except pan/tilt, fixed frame rate (**40 Hz by default**),
  encoder angles synced with exposure
- Detector: runs on every frame, known latency, same-thread or in-order —
  this library only consumes its `(u, v, sigma)` output
- Output: smoothed state + an image-space **search ROI** that widens
  automatically while the detector is blind
- Header-only, Eigen3 only, `float`, zero heap allocation on the hot path

Pixel measurements plus constant-acceleration kinematics are *linear*, so the
core is a plain 6-state linear Kalman filter — no sigma points, no
policy dispatch, no mode mixing.  Camera rotation is a known deterministic
input (exposure-synced encoders), so it is stripped at the API boundary rather
than estimated.

## Quick start

```cpp
#include <kalman/tracker.hpp>

kalman::Config cfg;
cfg.pan_px_per_rad  = 110.0f;   // hand-measured (see "Pan/tilt gains" below)
cfg.tilt_px_per_rad = -110.0f;  // sign follows your encoder convention
kalman::Tracker tracker(cfg);

tracker.reset(first_detection, pan, tilt);   // seed; pan/tilt define the frame
                                              // false → non-finite seed rejected

// once per captured frame, in capture order:
const bool accepted = tracker.step(pan, tilt, detection);  // nullopt on a miss
if (!accepted) { /* gated outlier — state untouched */ }
if (tracker.lost()) { /* full-frame search until it re-seeds */ }

// outputs:
auto x  = tracker.position();                // stabilised px
auto v  = tracker.velocity();                // px/s
auto p  = tracker.predict(latency_s);        // const lead, e.g. detector latency
auto roi = tracker.roi(next_pan, next_tilt, cfg.dt);  // window for the next frame
```

A complete runnable version of the loop lives in
[`examples/track_loop.cpp`](examples/track_loop.cpp).

## How it works

**State** (stabilised pixel frame): `[u, v, vu, vv, au, av]`.
Process model: constant acceleration with white-jerk noise (`q_jerk`).
Measurement: `(u, v)` from your detector, linear, so the update is a closed
form KF step (Joseph form, float, fixed-size). The data-oriented core stores
only the state and the unique covariance terms; it does not construct or
multiply dense process matrices per frame. See the design and measurements
below; run `./build/benchmarks/hotpath_bench` on your target for your own numbers.

**Pan/tilt compensation.**  Pure rotation shifts image content by an amount
independent of depth, so with encoder angles the camera's contribution to
apparent motion is exact, known input.  The tracker works in a *stabilised*
frame fixed at the orientation passed to `reset()`/first `step()`:
detections are shifted in, ROIs are shifted out, and the filter never sees
gimbal motion — one tuning works at any slew rate.

**Clock.**  `step()` advances exactly one frame period (`Config::dt`,
default `1/40` s; pass an explicit `dt` for dropped frames).  Call it once per
frame with that frame's encoder angles, in capture order.  Detector latency
never enters the state — use the const `predict(horizon)` when you need the
estimate at wall-clock-now (servo lead, display, latency compensation).

**Gating and coasting.**  Each detection is tested against a chi-square
innovation gate (`gate_chi2`, 9.21 = 99 % for 2 DOF).  Rejections — false
positives, a bird crossing the frame — leave the state untouched and return
`false`.  Misses (`std::nullopt`) and rejections both advance the clock;
`coasting()` reports the condition and the covariance (hence the ROI) grows
until the detector recovers.

**Track loss.**  After `Config::max_coast_frames` consecutive frames without
an accepted detection (default 120 = 3 s at 40 Hz) the track is declared
`lost()`: the coasted state is no longer worth trusting, `step()` freezes and
only re-seeds on the next detection, and you should widen your search to
full-frame until it arrives.  Set `max_coast_frames <= 0` to coast forever
instead.

**Bad input never reaches the state.**  `Config` is sanitised at construction
(non-finite or out-of-range fields fall back to their defaults),
`reset()` refuses a non-finite seed, and frames with garbage encoder angles or
detection coordinates are dropped with the clock still advancing — a NaN
anywhere degrades to "no information applied", never to a poisoned filter.

## Data-oriented core

The layout follows the actual workload: **one target with two independent,
equally tuned axes**, not an arbitrary-dimensional filter or a fleet of targets.

- Six adjacent state floats, paired by derivative: `[u, v, vu, vv, au, av]`.
- One symmetric per-axis covariance stored as six floats:
  `pp, pv, pa, vv, va, aa`. Both axes have identical priors, process noise,
  detector variance, timing, and joint gate decisions, so their covariance
  blocks are always identical and all cross-axis entries are zero.
- Plain data and direct predict/update kernels. The innovation covariance is
  `(pp + sigma²)·I₂`, so gating and gain calculation need one scalar reciprocal,
  not a matrix factorisation. Gating remains a **joint 2-DOF** decision.
- The covariance update retains Joseph form, expanded over the six unique
  entries. No subtractive-update shortcut, heap traffic, runtime dispatch,
  fast-math, or architecture-specific intrinsics.
- `roi()` calculates only the future position and position variance, with one
  square root for the shared extent. `predict()` and `covariance()` expand to
  a dense matrix only because their output contract requests one.

This deliberately avoids a batch/SoA framework: there is no batch in this
library's single-target workload. Adding one would add machinery rather than
remove work. Correlated axes or axis-specific noise would require a different
covariance representation; they are not supported by the current model.

**API migration:** `covariance()` now returns a `Matrix6` snapshot **by value**,
not a live reference to internal storage. Use `auto p = tracker.covariance()`;
retain that matrix before taking Eigen blocks or other lazy expressions from
it. A saved snapshot does not change on subsequent `step()` calls. The generic
`detail::LinearFilter<N, M>` and its `Tracker::Filter` alias have been removed.
The normal `reset/step/position/velocity/state/predict/roi` interface is unchanged.

### Performance

Representative same-source before/after measurement on an AMD Ryzen 7 5800X,
GCC 15.2, CMake Release (`-O3 -DNDEBUG`), pinned to one logical CPU:

| Workload (`step` + `roi` + full `predict`) | Dense core | Packed core | Speedup |
|---|---:|---:|---:|
| Detection every frame | 399 ns | 57 ns | 7.0× |
| Misses and rejected outliers | 378 ns | 55 ns | 6.9× |
| Prediction-only coasting | 261 ns | 43 ns | 6.1× |
| `sizeof(Tracker)` | 272 bytes | 132 bytes | 51% smaller |

Each timing is the median of 15 batches of 10,000 frames after warmup. Inputs
are precomputed outside timing, identical between batches, and generated
independently of the estimate. A compiler barrier consumes complete prediction
and ROI outputs, including covariance. Accepted counts and final tracking error
are also checked: both cores accepted 10,000/10,000 tracked detections and
8,294/10,000 mixed detections, with final errors of 0.015 px and 0.020 px.
The coast case disables loss detection to measure prediction, not the frozen
lost-track shortcut. These are CPU microbenchmarks, not detector/pipeline
latency promises; rerun on the deployment hardware (especially ARM).

## Pan/tilt gains (no intrinsics required)

You need one measured number per axis: pixels of image shift per radian of
encoder movement.  Point the camera at a static scene, sweep the encoder by a
known angle, divide the observed pixel displacement:

```
pan_gain  = Δu / Δpan     tilt_gain = Δv / Δtilt      (radians, sign included)
```

Set `Config::pan_px_per_rad` / `tilt_px_per_rad`; `0` (the default) disables
compensation.  If calibration intrinsics become available later, an exact ray
projection can replace the small-angle model behind the same interface.

Encoder angles are folded into a *continuous* axis every frame, so a gimbal
whose absolute encoder wraps at ±π (they all do) does not inject a 2π·gain
pixel jump into the filter at the seam — the usual way a hand-rolled
stabiliser loses its track.  Just keep calling `step()` with the raw angles;
the unwrap is internal.

## Tuning

| Knob | Default | Meaning |
|---|---|---|
| `dt` | `1/40` s | Nominal frame period |
| `q_jerk` | `2e5` px²/s⁵ | The one that matters: the acceleration can change by `√(q·T)` px/s² within `T` seconds, so `√(q·0.5) ≈ 316` px/s² over half a second. Raise for sharper maneuvers, lower for a smoother estimate |
| `detection_sigma` | 5 px | Assumed detector noise when `Detection::sigma` is unusable (≤ 0 or non-finite) |
| `gate_chi2` | 9.21 | Innovation gate (χ² quantile, 2 DOF) |
| `init_pos/vel/acc_sigma` | 30 px / 500 px/s / 1500 px/s² | `reset()` priors |
| `roi_k_sigma`, `roi_min_half_extent` | 3, 40 px | ROI = `max(k·σ, floor)` per axis — set the floor to at least your target's box size |
| `pan/tilt_px_per_rad` | 0 | Compensation gains, above |
| `max_coast_frames` | 120 | Misses before `lost()`; ≤ 0 disables loss detection |

Pass a per-detection `sigma` (from detector confidence) whenever you have
one: the gate and the update both use it.

## Integration contract

1. `reset(det, pan, tilt)` once, or just call `step()` — the first detection
   seeds the track and sets the stabilised-frame reference.  `reset` returns
   `false` (leaving the tracker untouched) if the seed or the encoder angles
   are not finite.
2. Every frame, in capture order: `step(pan, tilt, det)` — encoder angles from
   *that frame's exposure*, `nullopt` when the detector misses.  Feed the
   encoder every frame even on misses: the ±π unwrap needs every sample.
3. `step` returns `true` iff a detection was accepted; a `false` return means
   gated out (or unseeded), never a crash.
4. Build your next ROI with `roi(next_pan, next_tilt, horizon)` — pass the
   next frame's commanded/known encoder angles if the gimbal is moving, and
   `horizon = dt` (or your detector latency, if the ROI is consumed later).
5. Check `lost()` each frame: once true, drop to a full-frame search — the
   tracker stays frozen until the next detection re-seeds it.
6. One `Tracker` per target; not thread-safe (one per capture thread is the
   intended use).  The `Config` is copied and sanitised at construction;
   build a new `Tracker` to retune.

## Using the library

Header-only: only Eigen3 ≥ 3.4 is required.

```cmake
include(FetchContent)
FetchContent_Declare(kalman
  GIT_REPOSITORY https://github.com/ioeftedal/kalman-cpp
  GIT_TAG <release-or-commit> GIT_SHALLOW TRUE)
FetchContent_MakeAvailable(kalman)
target_link_libraries(your_app PRIVATE kalman::kalman)
```

Or `find_package(kalman)` after `cmake --install` (or `nix build`), or vendor
as a submodule with `add_subdirectory`.  Apps/tests default to OFF when
kalman is consumed as a subproject.

## Building and testing

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
ctest --test-dir build --output-on-failure   # 40 tests
./build/examples/track_loop                  # runnable contract demo
./build/benchmarks/hotpath_bench             # per-frame cost on this machine
```

The suite covers: process-model structure (F/Q, Gramian quadrature),
convergence, Monte-Carlo NEES *and* NIS consistency, outlier gating, coast
behaviour and track loss, pan/tilt compensation including the ±π encoder wrap
seam, latency extrapolation, ROI growth, input validation (NaN/Inf seeds,
encoder angles, `dt`, `Config` sanitisation), comparison against a dense
double-precision Joseph reference with varying timing/noise and zero jerk,
the joint gate boundary, and a `--wrap=malloc` allocation-freedom check on
the hot path (with prediction and ROI outputs consumed).

To smoke-test an install, `cmake --install` to a prefix and configure
`tests/package_consumer/` against it with `-DCMAKE_CXX_STANDARD=14` — it must
still compile as C++17, proving the exported target propagates `cxx_std_17`.

Options: `KALMAN_BUILD_TESTS` (ON top-level, OFF as subproject),
`KALMAN_INSTALL` (package export).

## Layout

```
include/kalman/
  tracker.hpp              # the entire public API: Config, Detection, Tracker
  detail/
    kf.hpp                 # packed filter data + direct CA/Joseph kernels
    ca_model.hpp           # analytic F/Q matrices for reference and validation
    stabilize.hpp          # pan/tilt ↔ pixel shift, ±π encoder unwrap
examples/track_loop.cpp     # 40 Hz loop contract demo
benchmarks/hotpath_bench.cpp # per-frame cost + tracking-error self-check
tests/                      # GTest suite incl. NEES/NIS, wrap-seam, no-alloc
tests/package_consumer/     # installed-package find_package smoke test
```

## References

- Bar-Shalom, Y., Li, X.R. and Kirubarajan, T., *Estimation with Applications
  to Tracking and Navigation*, 2001 (CA model, gating, NEES consistency).
- Julier, S.J. and Uhlmann, J.K., "A new extension of the Kalman filter to
  nonlinear systems", *Proc. SPIE*, 1997 (not used here — kept as the
  justification for *not* using a UKF on a linear model).
