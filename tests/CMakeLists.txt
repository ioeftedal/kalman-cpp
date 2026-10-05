add_executable(
  ukf_test
  core/imm_test.cpp core/alloc_counter_test.cpp helpers/alloc_counter.cpp
  models/pan_tilt_test.cpp models/pixel_model_test.cpp
  models/world_frame_test.cpp)
target_include_directories(ukf_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(ukf_test kalman GTest::gtest_main)
kalman_enable_warnings(ukf_test)
include(GoogleTest)
gtest_discover_tests(ukf_test)

# Allocation-freedom tests for the filter hot path. Eigen's aligned_malloc calls
# std::malloc directly rather than going through global operator new, so this
# target interposes the malloc family with the linker's --wrap mechanism.
# Without that, a Dynamic-typed Eigen matrix in the update path would be an
# invisible per-frame heap allocation.
add_executable(imm_noalloc_test core/imm_noalloc_test.cpp
                                helpers/alloc_counter.cpp)
target_include_directories(imm_noalloc_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_compile_definitions(imm_noalloc_test PRIVATE KALMAN_TEST_WRAP_MALLOC)
target_link_libraries(
  imm_noalloc_test
  kalman
  GTest::gtest_main
  "-Wl,--wrap=malloc"
  "-Wl,--wrap=free"
  "-Wl,--wrap=calloc"
  "-Wl,--wrap=realloc")
kalman_enable_warnings(imm_noalloc_test)
gtest_discover_tests(imm_noalloc_test)

add_executable(
  global_motion_test
  apps/global_motion_test.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/../apps/video_track/global_motion.cpp)
target_include_directories(
  global_motion_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../apps/video_track)
target_link_libraries(global_motion_test kalman ${OpenCV_LIBS}
                      GTest::gtest_main)
kalman_enable_warnings(global_motion_test)
gtest_discover_tests(global_motion_test)

add_executable(
  appearance_test
  apps/appearance_test.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/../apps/video_track/appearance.cpp)
target_include_directories(
  appearance_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../apps/video_track)
target_link_libraries(appearance_test kalman ${OpenCV_LIBS} GTest::gtest_main)
kalman_enable_warnings(appearance_test)
gtest_discover_tests(appearance_test)
