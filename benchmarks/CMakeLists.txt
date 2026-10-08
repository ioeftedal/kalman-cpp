# Hot-path microbenchmark.  Hand-rolled (steady_clock + median of batches)
# rather than Google Benchmark: the library has no third-party deps and this
# only needs to answer "how long does one frame's worth of filter work take on
# this machine", which is exactly what the README claim is based on.
add_executable(hotpath_bench hotpath_bench.cpp)
target_link_libraries(hotpath_bench kalman)
kalman_enable_warnings(hotpath_bench)
