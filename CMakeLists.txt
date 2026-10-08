cmake_minimum_required(VERSION 3.25)
project(
  kalman
  LANGUAGES CXX
  VERSION 0.2.0)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Component defaults: everything for a top-level (developer) build; tests are
# OFF when kalman is consumed via add_subdirectory / FetchContent so consumers
# are not forced to provide GTest.  Explicit -D overrides always win.
if(PROJECT_IS_TOP_LEVEL)
  set(KALMAN_DEFAULT_BUILD_TESTS ON)
else()
  set(KALMAN_DEFAULT_BUILD_TESTS OFF)
endif()

option(KALMAN_BUILD_TESTS "Build the test suite" ${KALMAN_DEFAULT_BUILD_TESTS})
option(KALMAN_INSTALL
       "Generate install/export rules for the kalman CMake package"
       ${PROJECT_IS_TOP_LEVEL})

if(KALMAN_BUILD_TESTS)
  enable_testing()
endif()

# Shared warning policy for every compiled target.
function(kalman_enable_warnings target)
  target_compile_options(${target} PRIVATE -Wall -Wextra -Wpedantic -Werror
                                           ${ARGN})
endfunction()

find_package(Eigen3 3.4 REQUIRED NO_MODULE)

# Header-only library: the whole public API is include/kalman/tracker.hpp.
add_library(kalman INTERFACE)
add_library(kalman::kalman ALIAS kalman)

target_include_directories(
  kalman INTERFACE $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
                   $<INSTALL_INTERFACE:include>)
target_link_libraries(kalman INTERFACE Eigen3::Eigen)
# Public headers include <optional>: consumers must compile as C++17. Without
# this, a C++14 consumer gets an unrelated hard error deep inside the standard
# library instead of a clear "kalman requires cxx_std_17".
target_compile_features(kalman INTERFACE cxx_std_17)

if(PROJECT_IS_TOP_LEVEL)
  add_subdirectory(examples)
  add_subdirectory(benchmarks)
endif()

if(KALMAN_BUILD_TESTS)
  # 1.14 is what current distros ship; nothing here needs a newer gtest and
  # requiring 1.17 would force a from-source build on Ubuntu.
  find_package(GTest 1.14 REQUIRED)
  add_subdirectory(tests)
endif()

# ---------------------------------------------------------------------------
# Installation / CMake package export (find_package(kalman)).
# ---------------------------------------------------------------------------
if(NOT KALMAN_INSTALL)
  return()
endif()

include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

set(KALMAN_CMAKE_INSTALL_DIR
    "${CMAKE_INSTALL_LIBDIR}/cmake/kalman"
    CACHE STRING "Install location for kalman's CMake package files")

install(TARGETS kalman EXPORT kalmanTargets)

install(
  DIRECTORY "${PROJECT_SOURCE_DIR}/include/kalman"
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
  FILES_MATCHING
  PATTERN "*.hpp")

install(
  EXPORT kalmanTargets
  FILE kalmanTargets.cmake
  NAMESPACE kalman::
  DESTINATION ${KALMAN_CMAKE_INSTALL_DIR})

write_basic_package_version_file(
  "${CMAKE_CURRENT_BINARY_DIR}/kalmanConfigVersion.cmake"
  VERSION ${PROJECT_VERSION}
  COMPATIBILITY SameMajorVersion ARCH_INDEPENDENT)

file(
  WRITE "${CMAKE_CURRENT_BINARY_DIR}/kalmanConfig.cmake"
  "include(CMakeFindDependencyMacro)\nfind_dependency(Eigen3 3.4 NO_MODULE)\ninclude(\"\${CMAKE_CURRENT_LIST_DIR}/kalmanTargets.cmake\")\n"
)

install(FILES "${CMAKE_CURRENT_BINARY_DIR}/kalmanConfig.cmake"
              "${CMAKE_CURRENT_BINARY_DIR}/kalmanConfigVersion.cmake"
        DESTINATION ${KALMAN_CMAKE_INSTALL_DIR})
