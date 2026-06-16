add_executable(simulator simulator/main.cpp)
target_link_libraries(simulator kalman)
target_compile_options(simulator PRIVATE -Wall -Wextra -Wpedantic -Werror)

add_executable(drone_sim drone_sim/main.cpp)
target_link_libraries(drone_sim kalman)
target_compile_options(drone_sim PRIVATE -Wall -Wextra -Wpedantic -Werror)

add_subdirectory(video_track)
