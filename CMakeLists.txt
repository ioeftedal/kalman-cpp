cmake_minimum_required(VERSION 3.25)
project(kalman LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

enable_testing()

find_package(Eigen3 3.4 REQUIRED NO_MODULE)
find_package(OpenCV 4 REQUIRED COMPONENTS core imgproc imgcodecs videoio video highgui dnn)

add_subdirectory(src)

add_subdirectory(apps)

find_package(GTest 1.17 REQUIRED)
add_subdirectory(tests)

find_package(benchmark 1.9 REQUIRED)
add_subdirectory(benchmarks)
