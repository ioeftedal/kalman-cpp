# Standalone smoke test for the *installed* package — deliberately not part of
# the main build (it needs `cmake --install` first).  Configure it against an
# install prefix with CMAKE_CXX_STANDARD=14: if the exported target ever stops
# declaring cxx_std_17, <optional> fails to compile here instead of in a user's
# first build.
cmake_minimum_required(VERSION 3.25)
project(kalman_consumer LANGUAGES CXX)

find_package(kalman REQUIRED)

add_executable(consumer main.cpp)
target_link_libraries(consumer PRIVATE kalman::kalman)
