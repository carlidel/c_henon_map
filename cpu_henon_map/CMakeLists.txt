cmake_minimum_required(VERSION 3.11 FATAL_ERROR)
# call with -A x64 on windows
project(all_henon_map)
find_package(OpenMP REQUIRED)
add_subdirectory(pybind11)

include_directories(c:/thrust)
set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS} -DTHRUST_DEVICE_SYSTEM=THRUST_DEVICE_SYSTEM_OMP")
set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")

add_library(henon_map STATIC
    ${PROJECT_SOURCE_DIR}/src/c_henon_map.cpp
    ${PROJECT_SOURCE_DIR}/src/c_henon_map.h
)

pybind11_add_module(cpu_henon_map ${PROJECT_SOURCE_DIR}/src/bindings.cpp)
target_link_libraries(cpu_henon_map PRIVATE henon_map)