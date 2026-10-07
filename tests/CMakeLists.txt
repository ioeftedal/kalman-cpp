add_executable(kalman_test ca_model_test.cpp stabilize_test.cpp
                           tracker_test.cpp)
target_include_directories(kalman_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(kalman_test kalman GTest::gtest_main)
kalman_enable_warnings(kalman_test)
include(GoogleTest)
gtest_discover_tests(kalman_test)

# Allocation-freedom test for the filter hot path. Eigen's aligned_malloc calls
# std::malloc directly rather than going through global operator new, so this
# target interposes the malloc family with the linker's --wrap mechanism;
# without it, an accidental heap use in step/predict/roi would be invisible.
add_executable(kalman_noalloc_test noalloc_test.cpp helpers/alloc_counter.cpp)
target_include_directories(kalman_noalloc_test
                           PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_compile_definitions(kalman_noalloc_test PRIVATE KALMAN_TEST_WRAP_MALLOC)
target_link_libraries(
  kalman_noalloc_test
  kalman
  GTest::gtest_main
  "-Wl,--wrap=malloc"
  "-Wl,--wrap=free"
  "-Wl,--wrap=calloc"
  "-Wl,--wrap=realloc")
kalman_enable_warnings(kalman_noalloc_test)
gtest_discover_tests(kalman_noalloc_test)
