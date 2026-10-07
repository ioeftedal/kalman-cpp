# The live-loop contract demo: build it top-level only, it documents the API far
# better than prose does.
add_executable(track_loop track_loop.cpp)
target_link_libraries(track_loop kalman)
kalman_enable_warnings(track_loop)
