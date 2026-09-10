/// @file
/// @brief C++17 drop-in replacement for the C++20 `<numbers>` constants.

#pragma once

namespace kalman {
namespace constants {

/// π as a `double`.
constexpr double pi = 3.141592653589793238462643383279502884L;

/// π as a value of floating-point type `T`.
template <typename T>
constexpr T pi_v = static_cast<T>(pi);

}  // namespace constants
}  // namespace kalman