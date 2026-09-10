#pragma once

#include <type_traits>

namespace kalman {

namespace detail {

template <typename Policy, typename = void>
struct is_kalman_policy_impl : std::false_type {};

template <typename Policy>
struct is_kalman_policy_impl<Policy,
                             std::void_t<typename Policy::ModelTag,
                                         decltype(static_cast<bool>(Policy::isMeasurementLinear))>>
    : std::true_type {};

}  // namespace detail

/// C++17 trait equivalent of the C++20 `KalmanPolicy` concept: a policy is
/// valid if it exposes `ModelTag` and a `isMeasurementLinear` that is
/// contextually convertible to `bool`.
template <typename Policy>
struct is_kalman_policy : detail::is_kalman_policy_impl<Policy> {};

template <typename Policy>
inline constexpr bool is_kalman_policy_v = is_kalman_policy<Policy>::value;

}  // namespace kalman