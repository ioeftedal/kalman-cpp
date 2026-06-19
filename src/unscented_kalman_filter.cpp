/// @file
/// @brief Explicit template instantiations for the UnscentedKalmanFilter.

#include "kalman/core/filter.hpp"
#include "kalman/models/pan_tilt.hpp"
#include "kalman/models/pixel.hpp"
#include "kalman/models/world_frame.hpp"

namespace kalman {

// -----------------------------------------------------------------------
//  3-D pan/tilt bearing-only models
// -----------------------------------------------------------------------
// 9-state CA: [x, y, z, vx, vy, vz, ax, ay, az]
template class UnscentedKalmanFilter<double, 9, 2, PanTiltPolicy>;
template class UnscentedKalmanFilter<float, 9, 2, PanTiltPolicy>;
// 6-state CV: [x, y, z, vx, vy, vz]
template class UnscentedKalmanFilter<double, 6, 2, PanTiltPolicy>;

// -----------------------------------------------------------------------
//  World-frame pan/tilt models
// -----------------------------------------------------------------------
template class UnscentedKalmanFilter<double, 9, 2, WorldFramePanTiltPolicy>;
template class UnscentedKalmanFilter<float, 9, 2, WorldFramePanTiltPolicy>;
template class UnscentedKalmanFilter<double, 6, 2, WorldFramePanTiltPolicy>;

// -----------------------------------------------------------------------
//  Pixel-space 2-D CV model (4-state)
//  State: [u, v, vu, vv]
// -----------------------------------------------------------------------
template class UnscentedKalmanFilter<double, 4, 2, PixelPolicy>;
template class UnscentedKalmanFilter<float, 4, 2, PixelPolicy>;

// -----------------------------------------------------------------------
//  Pixel-space 2-D CA model (6-state)
//  State: [u, v, vu, vv, au, av]
// -----------------------------------------------------------------------
template class UnscentedKalmanFilter<double, 6, 2, PixelCAPolicy>;
template class UnscentedKalmanFilter<float, 6, 2, PixelCAPolicy>;

}  // namespace kalman
