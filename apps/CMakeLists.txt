if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/simulator/main.cpp")
  add_executable(simulator simulator/main.cpp)
  target_link_libraries(simulator kalman)
  target_compile_options(simulator PRIVATE -Wall -Wextra -Wpedantic -Werror)
endif()

if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/drone_sim/main.cpp")
  add_executable(drone_sim drone_sim/main.cpp)
  target_link_libraries(drone_sim kalman)
  target_compile_options(drone_sim PRIVATE -Wall -Wextra -Wpedantic -Werror)
endif()

add_subdirectory(video_track)
