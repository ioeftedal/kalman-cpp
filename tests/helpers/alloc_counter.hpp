/// @file
/// @brief Global heap-allocation counters used to assert that the filter hot
///        path performs no dynamic allocation.
///
/// Real-time consumers run these filters inside a fixed-period control loop
/// with a hard deadline and an explicit "no heap allocation on the hot path"
/// budget. These helpers measure allocations so a test can bracket the region
/// of interest and assert the delta is zero.
///
/// Two levels of interception are needed, because no single hook sees
/// everything:
///
///  1. Global `operator new` / `operator delete` replacement. Always active.
///     Catches standard-library containers (`std::vector`, `std::string`,
///     `std::deque`, ...) but *not* Eigen: Eigen's `aligned_malloc` calls
///     `std::malloc` directly.
///  2. `malloc` / `calloc` / `realloc` interposition via the linker's
///     `--wrap` mechanism, compiled in only when `KALMAN_TEST_WRAP_MALLOC` is
///     defined. This is the only way to observe Eigen-internal allocation.
///
/// A target that defines `KALMAN_TEST_WRAP_MALLOC` must also be linked with
/// `-Wl,--wrap=malloc -Wl,--wrap=free -Wl,--wrap=calloc -Wl,--wrap=realloc`;
/// `tests/CMakeLists.txt` sets this up for the dedicated target.
///
/// Both counters feed into `count()`, and both only ever increase, so the
/// "delta == 0" assertions are independent of activity elsewhere in the
/// process (test framework, allocator bookkeeping, etc.).

#pragma once

#include <cstddef>

namespace alloc_counter {

/// Allocations observed since the last `reset()`, across every hooked level.
[[nodiscard]] std::size_t count() noexcept;

/// Reset the counter to zero.
void reset() noexcept;

/// Allocation delta over the enclosing scope.
class Scope {
 public:
  Scope() noexcept : start_(count()) {}
  [[nodiscard]] std::size_t delta() const noexcept { return count() - start_; }

 private:
  std::size_t start_;
};

}  // namespace alloc_counter