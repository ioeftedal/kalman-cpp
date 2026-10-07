/// @file
/// @brief Counting replacements for the global allocation functions.
///
/// `operator new` is replaced unconditionally. `malloc`/`calloc`/`realloc`
/// are additionally interposed when `KALMAN_TEST_WRAP_MALLOC` is defined,
/// because Eigen's `aligned_malloc` bypasses `operator new` and calls
/// `std::malloc` directly.
///
/// Deallocation is never counted: the invariant under test is "no allocation on
/// the hot path", and a free can only follow an allocation, so counting
/// acquisitions alone is both sufficient and immune to allocator caching.

#include "helpers/alloc_counter.hpp"

#include <atomic>
#include <cstddef>
#include <cstdlib>
#include <new>

namespace {

std::atomic<std::size_t> g_new_count{0};
#if defined(KALMAN_TEST_WRAP_MALLOC)
std::atomic<std::size_t> g_malloc_count{0};
#endif

void bumpNew() noexcept {
  g_new_count.fetch_add(1, std::memory_order_relaxed);
}

#if defined(KALMAN_TEST_WRAP_MALLOC)
void bumpMalloc() noexcept {
  g_malloc_count.fetch_add(1, std::memory_order_relaxed);
}
#endif

/// std::aligned_alloc requires the size to be a multiple of the alignment.
std::size_t roundUp(std::size_t n, std::size_t a) noexcept {
  return ((n + a - 1) / a) * a;
}

}  // namespace

namespace alloc_counter {

std::size_t count() noexcept {
#if defined(KALMAN_TEST_WRAP_MALLOC)
  return g_new_count.load(std::memory_order_relaxed) +
         g_malloc_count.load(std::memory_order_relaxed);
#else
  return g_new_count.load(std::memory_order_relaxed);
#endif
}

void reset() noexcept {
  g_new_count.store(0, std::memory_order_relaxed);
#if defined(KALMAN_TEST_WRAP_MALLOC)
  g_malloc_count.store(0, std::memory_order_relaxed);
#endif
}

}  // namespace alloc_counter

// ---------------------------------------------------------------------------
//  operator new / delete -- always intercepted
// ---------------------------------------------------------------------------

void* operator new(std::size_t n) {
  bumpNew();
  if (void* p = std::malloc(n != 0 ? n : 1)) {
    return p;
  }
  throw std::bad_alloc();
}

void* operator new[](std::size_t n) {
  return ::operator new(n);
}

void operator delete(void* p) noexcept {
  std::free(p);
}
void operator delete[](void* p) noexcept {
  std::free(p);
}
void operator delete(void* p, std::size_t) noexcept {
  std::free(p);
}
void operator delete[](void* p, std::size_t) noexcept {
  std::free(p);
}
void operator delete(void* p, const std::nothrow_t&) noexcept {
  std::free(p);
}
void operator delete[](void* p, const std::nothrow_t&) noexcept {
  std::free(p);
}

void* operator new(std::size_t n, std::align_val_t a) {
  bumpNew();
  const std::size_t align = static_cast<std::size_t>(a);
  if (void* p = std::aligned_alloc(align, roundUp(n != 0 ? n : 1, align))) {
    return p;
  }
  throw std::bad_alloc();
}

void* operator new[](std::size_t n, std::align_val_t a) {
  return ::operator new(n, a);
}

void operator delete(void* p, std::align_val_t) noexcept {
  std::free(p);
}
void operator delete[](void* p, std::align_val_t) noexcept {
  std::free(p);
}
void operator delete(void* p, std::size_t, std::align_val_t) noexcept {
  std::free(p);
}
void operator delete[](void* p, std::size_t, std::align_val_t) noexcept {
  std::free(p);
}
void operator delete(void* p, std::align_val_t,
                     const std::nothrow_t&) noexcept {
  std::free(p);
}
void operator delete[](void* p, std::align_val_t,
                       const std::nothrow_t&) noexcept {
  std::free(p);
}

// ---------------------------------------------------------------------------
//  malloc family interposition -- only when KALMAN_TEST_WRAP_MALLOC is defined
// ---------------------------------------------------------------------------

#if defined(KALMAN_TEST_WRAP_MALLOC)

extern "C" {
void* __real_malloc(std::size_t);
void __real_free(void*);
void* __real_calloc(std::size_t, std::size_t);
void* __real_realloc(void*, std::size_t);
}

extern "C" void* __wrap_malloc(std::size_t n) {
  bumpMalloc();
  return __real_malloc(n);
}

extern "C" void __wrap_free(void* p) {
  __real_free(p);
}

extern "C" void* __wrap_calloc(std::size_t n, std::size_t s) {
  bumpMalloc();
  return __real_calloc(n, s);
}

extern "C" void* __wrap_realloc(void* p, std::size_t n) {
  bumpMalloc();
  return __real_realloc(p, n);
}

#endif  // KALMAN_TEST_WRAP_MALLOC