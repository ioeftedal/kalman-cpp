#pragma once

#include <concepts>
#include <type_traits>

namespace kalman {

template <typename Policy>
concept KalmanPolicy = requires {
  typename Policy::ModelTag;
  { Policy::isMeasurementLinear } -> std::convertible_to<bool>;
};

}  // namespace kalman
