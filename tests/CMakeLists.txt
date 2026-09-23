add_executable(ukf_test core/imm_test.cpp models/pan_tilt_test.cpp
                        models/pixel_model_test.cpp models/world_frame_test.cpp)
target_link_libraries(ukf_test kalman GTest::gtest_main)
kalman_enable_warnings(ukf_test)
include(GoogleTest)
gtest_discover_tests(ukf_test)

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
