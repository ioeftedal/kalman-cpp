/// @file
/// @brief Explicit template instantiations for the UnscentedKalmanFilter.
///
/// Pre-compiling the commonly-used template specialisations avoids
/// re-instantiation overhead in translation units that include the header.
/// Each `template class` directive forces the compiler to emit all member
/// function definitions for the given template arguments.

#include "kalman/unscented_kalman_filter.hpp"

#include "kalman/pixel_measurement_model.hpp"

namespace kalman {

// -----------------------------------------------------------------------
//  3-D pan/tilt bearing-only models (default PanTiltPolicy)
// -----------------------------------------------------------------------
// 9-state CA:  [x, y, z, vx, vy, vz, ax, ay, az]
template class UnscentedKalmanFilter<double, 9, 2>;
template class UnscentedKalmanFilter<float, 9, 2>;
// 6-state CV:  [x, y, z, vx, vy, vz]
template class UnscentedKalmanFilter<double, 6, 2>;

// -----------------------------------------------------------------------
//  Pixel-space 2-D CV model (4-state, PixelPolicy)
//  State: [u, v, vu, vv]
// -----------------------------------------------------------------------
template class UnscentedKalmanFilter<double, 4, 2, PixelPolicy>;
template class UnscentedKalmanFilter<float, 4, 2, PixelPolicy>;

// -----------------------------------------------------------------------
//  Pixel-space 2-D CA model (6-state, PixelCAPolicy)
//  State: [u, v, vu, vv, au, av]
// -----------------------------------------------------------------------
template class UnscentedKalmanFilter<double, 6, 2, PixelCAPolicy>;
template class UnscentedKalmanFilter<float, 6, 2, PixelCAPolicy>;

}  // namespace kalman
