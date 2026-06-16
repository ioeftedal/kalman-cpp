#pragma once

namespace kalman {

/// Primary template for process model dispatch.
///
/// Specialisations provide static `F(dt)` and `Q(dt, q_c)` member
/// functions for a given scalar type `T`, state dimension `N`, and
/// model tag.
///
/// The `Tag` parameter (default `void`) allows multiple process models
/// with the same state dimension to coexist at compile time.
template <typename T, int N, typename Tag = void>
struct ProcessModel;

}  // namespace kalman
