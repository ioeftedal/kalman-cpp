# Deep (MixFormerV2-S / ONNX Runtime) tracking backend. Disabled by default:
# point ONNXRUNTIME_DIR at an onnxruntime install that contains include + lib,
# or export ONNXRUNTIME_DIR, then configure with -DKALMAN_ENABLE_ONNXRUNTIME=ON.
option(KALMAN_ENABLE_ONNXRUNTIME "Build the ONNX Runtime deep-tracking backend"
       OFF)

set(DEEP_TRACK_SOURCES)

if(KALMAN_ENABLE_ONNXRUNTIME)
  set(ONNXRUNTIME_DIR
      ""
      CACHE PATH "onnxruntime install dir (contains include/ and lib/)")
  set(_ort_root "${ONNXRUNTIME_DIR}")
  if(NOT _ort_root)
    set(_ort_root "$ENV{ONNXRUNTIME_DIR}")
  endif()
  if(NOT _ort_root)
    message(
      FATAL_ERROR
        "KALMAN_ENABLE_ONNXRUNTIME=ON requires ONNXRUNTIME_DIR or the "
        "ONNXRUNTIME_DIR environment variable")
  endif()

  find_path(ONNXRUNTIME_INCLUDE_DIR onnxruntime_cxx_api.h
            HINTS "${_ort_root}/include")
  find_library(
    ONNXRUNTIME_LIBRARY
    NAMES onnxruntime
    HINTS "${_ort_root}/lib")

  if(ONNXRUNTIME_INCLUDE_DIR AND ONNXRUNTIME_LIBRARY)
    set(DEEP_TRACK_SOURCES deep_tracker.cpp)
    message(
      STATUS
        "video_track: deep tracking backend ON (onnxruntime ${ONNXRUNTIME_LIBRARY})"
    )
  else()
    message(
      WARNING
        "video_track: KALMAN_ENABLE_ONNXRUNTIME=ON but onnxruntime was not found "
        "under ${_ort_root}; building without the deep backend")
  endif()
endif()

add_executable(
  video_track main.cpp detector.cpp flow_tracker.cpp global_motion.cpp
              appearance.cpp ${DEEP_TRACK_SOURCES})
target_link_libraries(video_track kalman ${OpenCV_LIBS})
kalman_enable_warnings(video_track -Wno-c11-extensions)

if(DEEP_TRACK_SOURCES)
  target_compile_definitions(video_track PRIVATE KALMAN_ENABLE_DEEP=1)
  target_include_directories(video_track SYSTEM
                             PRIVATE "${ONNXRUNTIME_INCLUDE_DIR}")
  target_link_libraries(video_track "${ONNXRUNTIME_LIBRARY}")
  get_filename_component(_ort_lib_dir "${ONNXRUNTIME_LIBRARY}" DIRECTORY)
  target_link_options(video_track PRIVATE "-Wl,-rpath,${_ort_lib_dir}")
endif()

add_executable(video_track_eval eval.cpp)
kalman_enable_warnings(video_track_eval)

add_executable(video_track_annotate annotate.cpp)
target_link_libraries(video_track_annotate ${OpenCV_LIBS})
kalman_enable_warnings(video_track_annotate)
