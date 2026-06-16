add_library(kalman unscented_kalman_filter.cpp)
target_include_directories(kalman PUBLIC "${PROJECT_SOURCE_DIR}/include")
target_link_libraries(kalman PUBLIC Eigen3::Eigen)
target_compile_options(kalman PRIVATE -Wall -Wextra -Wpedantic -Werror)
