add_executable(video_track main.cpp detector.cpp)
target_link_libraries(video_track kalman
  ${OpenCV_LIBS})
target_compile_options(video_track PRIVATE -Wall -Wextra -Wpedantic -Werror
  -Wno-c11-extensions)
