cmake_minimum_required(VERSION 3.15.1)
project(vlm_stream)

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wl,--allow-shlib-undefined")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -Wl,--allow-shlib-undefined")

if(${CMAKE_VERSION} VERSION_GREATER "3.15.0")
  add_link_options("-Wl,-Bsymbolic")
endif()

set(TARGET_LIB_ARCH aarch64)
set(RKLLM_REPO $ENV{HOME}/rknn-llm)
set(MM_DEPLOY ${RKLLM_REPO}/examples/multimodal_model_demo/deploy)

# OpenCV
set(OpenCV_DIR ${MM_DEPLOY}/3rdparty/opencv/opencv-linux-aarch64/share/OpenCV)
find_package(OpenCV REQUIRED)

# RKNN runtime (for vision encoder)
set(RKNN_PATH ${MM_DEPLOY}/3rdparty/librknnrt)
set(LIBRKNNRT ${RKNN_PATH}/Linux/librknn_api/${TARGET_LIB_ARCH}/librknnrt.so)
set(LIBRKNNRT_INCLUDES ${RKNN_PATH}/Linux/librknn_api/include)

# RKLLM runtime (for LLM)
set(RKLLM_PATH ${RKLLM_REPO}/rkllm-runtime)
set(LIBRKLLMRT ${RKLLM_PATH}/Linux/librkllm_api/${TARGET_LIB_ARCH}/librkllmrt.so)
set(LIBRKLLMRT_INCLUDES ${RKLLM_PATH}/Linux/librkllm_api/include)

include_directories(${MM_DEPLOY}/src ${LIBRKNNRT_INCLUDES} ${LIBRKLLMRT_INCLUDES})

add_executable(vlm_stream vlm_stream.cpp ${MM_DEPLOY}/src/image_enc.cc)
target_link_libraries(vlm_stream PRIVATE ${OpenCV_LIBS} ${LIBRKNNRT} ${LIBRKLLMRT})
set_target_properties(vlm_stream PROPERTIES
    LINK_FLAGS "-Wl,--no-undefined,--version-script,${MM_DEPLOY}/c_export.map"
)
