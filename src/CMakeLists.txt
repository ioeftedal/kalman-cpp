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
# Installation
#
# `ImmFilter` and the pixel / pixel-CA / pixel-CT policies are header-only and
# are *not* in the static archive below (which holds only the explicit double
# UKF instantiations), so downstream consumers using a different scalar or an
# uncovered model must include the headers directly. The installed interface
# therefore ships both the archive and the headers.
#
# Export is skipped for Tracy-enabled builds on purpose. Tracy is attached to
# the target as a PUBLIC dependency, but it is normally provided by
# FetchContent, which produces a build-tree-only target that cannot appear in an
# export set -- `find_package(kalman)` would fail on the missing
# `Tracy::TracyClient` imported target. Install a Tracy-enabled build only after
# installing Tracy itself as a findable CMake package.
# ---------------------------------------------------------------------------

if(KALMAN_USE_TRACY)
  message(
    STATUS
      "kalman: Tracy enabled -- skipping install/export of the CMake package "
      "(Tracy::TracyClient is a build-tree-only target). Use "
      "-DKALMAN_USE_TRACY=OFF to install a consumable package.")
else()
  include(GNUInstallDirs)
  include(CMakePackageConfigHelpers)

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

  file(
    WRITE "${CMAKE_CURRENT_BINARY_DIR}/kalmanConfig.cmake"
    "include(CMakeFindDependencyMacro)\n"
    "find_dependency(Eigen3 3.4 NO_MODULE)\n"
    "include(\"\${CMAKE_CURRENT_LIST_DIR}/kalmanTargets.cmake\")\n")

  install(FILES "${CMAKE_CURRENT_BINARY_DIR}/kalmanConfig.cmake"
                "${CMAKE_CURRENT_BINARY_DIR}/kalmanConfigVersion.cmake"
          DESTINATION ${KALMAN_CMAKE_INSTALL_DIR})
endif()
