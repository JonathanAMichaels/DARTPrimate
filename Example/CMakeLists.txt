project("DARTPrimateExample")
cmake_minimum_required(VERSION 3.1.0)

if( NOT CMAKE_BUILD_TYPE )
   set( CMAKE_BUILD_TYPE Release )
endif()

add_definitions(-std=c++11)

find_path(DART_ROOT NAMES lib/libdart.so src/tracker.h HINTS ${PROJECT_SOURCE_DIR}/../)
message(STATUS "Found dart is at ${DART_ROOT}")
include_directories(${DART_ROOT}/src)
link_directories(${DART_ROOT}/lib)


set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${DART_ROOT}/CMakeModules/")

# find packages with modules
find_package(Pangolin REQUIRED)
find_package(PNG REQUIRED)
find_package(CUDA REQUIRED)

# find packages with pkg-config
find_package(PkgConfig)
pkg_check_modules(eigen3 REQUIRED eigen3)
pkg_check_modules(matheval REQUIRED libmatheval)

include_directories(  ${PNG_INCLUDE_DIRS})
link_libraries(       ${PNG_LIBRARIES})

include_directories(  ${Pangolin_INCLUDE_DIRS} )
link_directories(     ${Pangolin_LIBRARY_DIRS} )
link_libraries(       ${Pangolin_LIBRARIES}    )

include_directories(
    ${eigen3_INCLUDE_DIR}
    ${PROJECT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/src
)



##########################################################################
#   Build GPU or CPU-only executable depending on cuda                   #
##########################################################################

include_directories( ${CUDA_INCLUDE_DIRS} )
cuda_include_directories(${CMAKE_CURRENT_SOURCE_DIR})

set(CUDA_NVCC_FLAGS "-arch=sm_35" "--use_fast_math"  "-O3" "--ptxas-options=--verbose") # "-fmad=false" "-DTHRUST_DEVICE_BACKEND=THRUST_DEVICE_BACKEND_OMP"
add_definitions(-DCUDA_BUILD)

set(all_sources ${sources} ${cpu_sources} ${gpu_sources})
link_directories( ${CUDA_TOOLKIT_ROOT_DIR}/lib64/)

cuda_add_executable(
    #executable
    DARTPrimate_Example
    #sources
    ${all_sources}
    #main
    DARTPrimate_Example.cpp
)
target_link_libraries(DARTPrimate_Example dart)


