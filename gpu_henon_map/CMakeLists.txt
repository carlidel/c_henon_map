cmake_minimum_required(VERSION 3.11 FATAL_ERROR)
# call with -A x64 on windows
project(all_henon_map LANGUAGES CXX CUDA)
add_subdirectory(pybind11)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fpic")
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
include_directories(CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES)

add_library(g_henon_map STATIC
    src/c_henon_map.cu
    src/c_henon_map.h
)

target_link_libraries(g_henon_map PUBLIC ${CUDA_LIBRARIES})
target_compile_features(g_henon_map PUBLIC cxx_std_11)
#set_target_properties( henon_map PROPERTIES CUDA_SEPARABLE_COMPILATION ON )

pybind11_add_module(gpu_henon_map_core src/gpu_bindings.cpp)
target_link_libraries(gpu_henon_map_core PUBLIC g_henon_map)