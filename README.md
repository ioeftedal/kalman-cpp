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

// once per captured frame, in capture order:
tracker.step(pan, tilt, detection);          // std::nullopt on a detector miss
if (!accepted) { /* gated outlier — state untouched */ }

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
form KF step (Joseph form, float, fixed-size — a few microseconds on an ARM
core).

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

## Tuning

| Knob | Default | Meaning |
|---|---|---|
| `dt` | `1/40` s | Nominal frame period |
| `q_jerk` | `2e5` px²/s⁵ | The one that matters: acceleration can change by `√(q·0.5)` px within ~0.5 s. Raise for sharper maneuvers, lower for a smoother estimate |
| `detection_sigma` | 5 px | Assumed detector noise when `Detection::sigma ≤ 0` |
| `gate_chi2` | 9.21 | Innovation gate (χ² quantile, 2 DOF) |
| `init_pos/vel/acc_sigma` | 30 px / 500 px/s / 1500 px/s² | `reset()` priors |
| `roi_k_sigma`, `roi_min_half_extent` | 3, 40 px | ROI = `max(k·σ, floor)` per axis — set the floor to at least your target's box size |
| `pan/tilt_px_per_rad` | 0 | Compensation gains, above |

Pass a per-detection `sigma` (from detector confidence) whenever you have
one: the gate and the update both use it.

## Integration contract

1. `reset(det, pan, tilt)` once, or just call `step()` — the first detection
   seeds the track and sets the stabilised-frame reference.
2. Every frame, in capture order: `step(pan, tilt, det)` — encoder angles from
   *that frame's exposure*, `nullopt` when the detector misses.
3. `step` returns `true` iff a detection was accepted; a `false` return means
   gated out (or unseeded), never a crash.
4. Build your next ROI with `roi(next_pan, next_tilt, horizon)` — pass the
   next frame's commanded/known encoder angles if the gimbal is moving, and
   `horizon = dt` (or your detector latency, if the ROI is consumed later).
5. One `Tracker` per target; not thread-safe (one per capture thread is the
   intended use).

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
ctest --test-dir build --output-on-failure   # 23 tests
./build/examples/track_loop                  # runnable contract demo
```

The suite covers: process-model structure (F/Q, Gramian quadrature),
convergence, Monte-Carlo NEES consistency, outlier gating, coast behaviour,
pan/tilt compensation, latency extrapolation, ROI growth, and a
`--wrap=malloc` allocation-freedom check on the hot path.

Options: `KALMAN_BUILD_TESTS` (ON top-level, OFF as subproject),
`KALMAN_INSTALL` (package export).

## Layout

```
include/kalman/
  tracker.hpp              # the entire public API: Config, Detection, Tracker
  detail/
    kf.hpp                 # linear KF core: predict + gated Joseph update
    ca_model.hpp           # CA transition F(dt), white-jerk noise Q(dt, q)
    stabilize.hpp          # pan/tilt ↔ pixel shift, reference frame
examples/track_loop.cpp     # 40 Hz loop contract demo
tests/                      # GTest suite incl. NEES + no-alloc
```

## References

- Bar-Shalom, Y., Li, X.R. and Kirubarajan, T., *Estimation with Applications
  to Tracking and Navigation*, 2001 (CA model, gating, NEES consistency).
- Julier, S.J. and Uhlmann, J.K., "A new extension of the Kalman filter to
  nonlinear systems", *Proc. SPIE*, 1997 (not used here — kept as the
  justification for *not* using a UKF on a linear model).
