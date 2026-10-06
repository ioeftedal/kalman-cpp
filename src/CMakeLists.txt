add_library(kalman unscented_kalman_filter.cpp)
add_library(kalman::kalman ALIAS kalman)

target_include_directories(
  kalman PUBLIC $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
                $<INSTALL_INTERFACE:include>)
target_link_libraries(kalman PUBLIC Eigen3::Eigen)
kalman_enable_warnings(kalman)

if(KALMAN_USE_TRACY)
  target_compile_definitions(kalman PUBLIC TRACY_ENABLE)
  target_link_libraries(kalman PUBLIC Tracy::TracyClient)

  # Ghostty-style capture options.
  if(KALMAN_TRACY_ON_DEMAND)
    target_compile_definitions(kalman PUBLIC TRACY_ON_DEMAND)
  endif()
  if(KALMAN_TRACY_CALLSTACK GREATER 0)
    target_compile_definitions(kalman
                               PUBLIC TRACY_CALLSTACK=${KALMAN_TRACY_CALLSTACK})
  endif()
endif()

# ---------------------------------------------------------------------------
# Installation / CMake package export
#
# `ImmFilter` and the pixel / pixel-CA / pixel-CT policies are header-only and
# are *not* in the static archive below (which holds only the explicit `double`
# / `float` UKF instantiations), so downstream consumers using an uncovered
# model must include the headers directly. The installed interface therefore
# ships both the archive and the headers.
#
# Install/export is controlled by KALMAN_INSTALL (default ON for a top-level
# build, OFF when kalman is a subproject). One case still skips it: when Tracy
# was vendored via FetchContent, `Tracy::TracyClient` is a build-tree-only
# target that cannot appear in an export set -- `find_package(kalman)` would
# fail on the missing imported target. Build with -DKALMAN_USE_TRACY=OFF, or
# against a findable Tracy package (e.g. nixpkgs), to install a consumable
# package.
# ---------------------------------------------------------------------------

include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

option(KALMAN_INSTALL
       "Generate install/export rules for the kalman CMake package"
       ${PROJECT_IS_TOP_LEVEL})

if(NOT KALMAN_INSTALL)
  return()
endif()

if(KALMAN_USE_TRACY AND KALMAN_TRACY_FETCHED)
  message(
    STATUS
      "kalman: Tracy was vendored via FetchContent (build-tree-only "
      "Tracy::TracyClient) -- skipping install/export of the CMake package. "
      "Build with -DKALMAN_USE_TRACY=OFF, or against a findable Tracy package, "
      "to install a consumable package.")
  return()
endif()

set(KALMAN_CMAKE_INSTALL_DIR
    "${CMAKE_INSTALL_LIBDIR}/cmake/kalman"
    CACHE STRING "Install location for kalman's CMake package files")

install(
  TARGETS kalman
  EXPORT kalmanTargets
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
  INCLUDES
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

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

# The exported target links Eigen3 (always) and, when a findable Tracy package
# was used, Tracy::TracyClient -- so the config must resolve both before loading
# the targets file.
set(KALMAN_CONFIG_DEPS "include(CMakeFindDependencyMacro)\n")
if(KALMAN_USE_TRACY)
  string(APPEND KALMAN_CONFIG_DEPS "find_dependency(Tracy)\n")
endif()
string(APPEND KALMAN_CONFIG_DEPS "find_dependency(Eigen3 3.4 NO_MODULE)\n"
       "include(\"\${CMAKE_CURRENT_LIST_DIR}/kalmanTargets.cmake\")\n")

file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/kalmanConfig.cmake"
     "${KALMAN_CONFIG_DEPS}")

install(FILES "${CMAKE_CURRENT_BINARY_DIR}/kalmanConfig.cmake"
              "${CMAKE_CURRENT_BINARY_DIR}/kalmanConfigVersion.cmake"
        DESTINATION ${KALMAN_CMAKE_INSTALL_DIR})
