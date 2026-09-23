add_library(kalman unscented_kalman_filter.cpp)
target_include_directories(kalman PUBLIC "${PROJECT_SOURCE_DIR}/include")
target_link_libraries(kalman PUBLIC Eigen3::Eigen)
kalman_enable_warnings(kalman)

if(KALMAN_USE_TRACY)
  target_compile_definitions(kalman PUBLIC TRACY_ENABLE)
  target_link_libraries(kalman PUBLIC Tracy::TracyClient)

  # Ghostty-style capture options.
  if(KALMAN_TRACY_ON_DEMAND)
    target_compile_definitions(kalman PUBLIC TRACY_ON_DEMAND)
  endif()
  if(KALMAN_TRACY_CALLSTACK GREATER 0)
    target_compile_definitions(kalman
                               PUBLIC TRACY_CALLSTACK=${KALMAN_TRACY_CALLSTACK})
  endif()
endif()
