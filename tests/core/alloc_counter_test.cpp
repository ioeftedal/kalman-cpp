/// @file
/// @brief Canary tests proving the allocation counters actually observe heap
///        traffic.
///
/// Every other allocation-freedom test in the suite asserts a delta of zero.
/// That assertion is only meaningful if these canaries pass, so they live in
/// the same binary as the tests that depend on them. If a counter silently
/// stops counting -- a linker change dropping the replacement operators, a
/// misconfigured `--wrap`, or aggressive dead-code elimination removing the very
/// allocation under observation -- these fail loudly instead of letting the
/// real tests pass vacuously.

#include "helpers/alloc_counter.hpp"

#include <gtest/gtest.h>

#include <cstddef>
#include <deque>
#include <string>
#include <vector>

namespace {

// A `new`/`delete` pair whose result is unused is a legal candidate for
// elision, so the pointer is published to a volatile global to force the
// allocation to actually happen.
volatile void* g_sink = nullptr;

}  // namespace

TEST(AllocCounter, DetectsPlainNew) {
  alloc_counter::reset();
  const alloc_counter::Scope scope;
  auto* p = new double(1.0);
  g_sink = p;
  EXPECT_EQ(scope.delta(), 1U) << "counter did not observe a plain heap allocation";
  EXPECT_DOUBLE_EQ(*static_cast<double*>(const_cast<void*>(g_sink)), 1.0);
  delete p;
  g_sink = nullptr;
}

TEST(AllocCounter, DetectsArrayNew) {
  alloc_counter::reset();
  const alloc_counter::Scope scope;
  auto* block = new int[64];
  g_sink = block;
  EXPECT_GE(scope.delta(), 1U) << "counter did not observe an array allocation";
  delete[] block;
  g_sink = nullptr;
}

TEST(AllocCounter, DetectsStdVectorGrowth) {
  std::vector<int> v;
  v.reserve(16);
  alloc_counter::reset();
  const alloc_counter::Scope scope;
  v.resize(4096, 7);
  EXPECT_GE(scope.delta(), 1U) << "counter did not observe std::vector growth";
  EXPECT_EQ(v[4095], 7);
}

TEST(AllocCounter, DetectsStdDequeChurn) {
  // std::deque allocates through global operator new -- this is what the
  // IMM's adaptive-noise history used to use.
  std::deque<int> d;
  d.push_back(1);
  alloc_counter::reset();
  const alloc_counter::Scope scope;
  for (int i = 0; i < 200; ++i) {
    d.push_back(i);
    d.pop_front();
  }
  EXPECT_GE(scope.delta(), 1U) << "counter did not observe std::deque churn";
}

TEST(AllocCounter, DetectsStdStringGrowth) {
  std::string s;
  alloc_counter::reset();
  const alloc_counter::Scope scope;
  for (int i = 0; i < 64; ++i) {
    s += "kalman-alloc-probe";
  }
  EXPECT_GE(scope.delta(), 1U) << "counter did not observe std::string growth";
  EXPECT_FALSE(s.empty());
}