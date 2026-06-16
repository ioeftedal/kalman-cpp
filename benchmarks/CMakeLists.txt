add_executable(ukf_benchmark ukf_benchmark.cpp)
target_link_libraries(ukf_benchmark kalman benchmark::benchmark)
target_compile_options(ukf_benchmark PRIVATE -Wall -Wextra -Wpedantic -Werror)
