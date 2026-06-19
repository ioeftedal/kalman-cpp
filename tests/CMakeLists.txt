add_executable(ukf_test
  models/pan_tilt_test.cpp
  models/pixel_model_test.cpp
  models/world_frame_test.cpp
  models/thermal_model_test.cpp
)
target_link_libraries(ukf_test kalman GTest::gtest_main)
target_compile_options(ukf_test PRIVATE -Wall -Wextra -Wpedantic -Werror)
include(GoogleTest)
gtest_discover_tests(ukf_test)
