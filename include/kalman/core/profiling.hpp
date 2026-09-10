/// @file
/// @brief Unified profiling instrumentation (Ghostty-style debug module).
///
/// Mirrors how Ghostty routes all instrumentation through a single debug
/// module: every macro below is either a thin alias to the underlying
/// profiler (Tracy) or compiles to a zero-cost no-op, so instrumentation
/// is compiled out of non-profiling builds without having to touch the
/// call sites.
///
/// The zone macros come in three flavours:
///   - `KALMAN_PROFILE_ZONE()`          auto-named from the source location
///                                      (the enclosing function name);
///   - `KALMAN_PROFILE_ZONE_NAMED(x)`   explicitly named (e.g. `UKF::predict`);
///   - `KALMAN_PROFILE_ZONE_NAMED_C`    named + colored per subsystem.
///
/// Frame marks are emitted from the application frame loop only; the
/// library itself never marks frames.
///
/// Enabling profiling requires the build to define `TRACY_ENABLE` for
/// the whole project (see CMake option `KALMAN_USE_TRACY`).  With Tracy
/// disabled, this header needs no Tracy headers and every macro compiles
/// to nothing under `-Wall -Wextra -Wpedantic -Werror`.

#pragma once

#include <cstdint>

namespace kalman::profiling {

/// Per-subsystem zone colors (RGB, upper byte unused), so the Tracy
/// timeline groups work together visually.
enum Color : std::uint32_t {
  ColorCore = 0xCC6600u,    // UKF / filter core
  ColorVision = 0x2299FFu,  // detection (YOLO)
  ColorMotion = 0x66CC00u,  // optical flow / KLT
  ColorApp = 0xFF4477u,     // application / HUD
};

}  // namespace kalman::profiling

#ifdef TRACY_ENABLE

#include <tracy/Tracy.hpp>

// Instrumentation zone, auto-named from the enclosing function.
#define KALMAN_PROFILE_ZONE() ZoneScoped
// Named instrumentation zone.
#define KALMAN_PROFILE_ZONE_NAMED(name) ZoneScopedN(name)
// Named + colored instrumentation zone (color is a `kalman::profiling::Color`).
#define KALMAN_PROFILE_ZONE_NAMED_C(name, color) ZoneScopedNC(name, color)
// Mark the end of a frame in the application frame loop.
#define KALMAN_PROFILE_FRAME() FrameMark
// Name the calling thread (do this once at thread start).
#define KALMAN_PROFILE_SET_THREAD_NAME(name) ::tracy::SetThreadName(name)
// Plot a double-precision metric over time.
#define KALMAN_PROFILE_PLOT(name, value) TracyPlot(name, value)

#else  // !TRACY_ENABLE — zero-cost no-ops.

#define KALMAN_PROFILE_ZONE() (void)0
#define KALMAN_PROFILE_ZONE_NAMED(name) (void)sizeof((name))
#define KALMAN_PROFILE_ZONE_NAMED_C(name, color) (void)sizeof((name), (color))
#define KALMAN_PROFILE_FRAME() (void)0
#define KALMAN_PROFILE_SET_THREAD_NAME(name) (void)sizeof((name))
#define KALMAN_PROFILE_PLOT(name, value) (void)sizeof((name)), (void)(value)

#endif  // TRACY_ENABLE