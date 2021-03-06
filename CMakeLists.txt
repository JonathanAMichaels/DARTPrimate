project("dart")
cmake_minimum_required(VERSION 3.1.0)

# build release if not explicitly told to build debug
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# turn on c++11
add_definitions(-std=c++11)


set(CC "61")

# set CMakeModule path
#set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${PROJECT_SOURCE_DIR}/CMakeModules/")

# find packages with modules
find_package(CUDA REQUIRED)
find_package(OpenGL REQUIRED QUIET)
find_package(GLUT REQUIRED)
find_package(GLEW REQUIRED)

# find packages with pkg-config
find_package(PkgConfig)
pkg_check_modules(eigen3 REQUIRED eigen3)
pkg_check_modules(matheval REQUIRED libmatheval)
pkg_check_modules(tinyxml REQUIRED tinyxml)
pkg_check_modules(assimp assimp)

include_directories(
    ${eigen3_INCLUDE_DIRS}
    ${OPENGL_INCLUDE_DIR}
    ${GLUT_INCLUDE_DIR}
    ${GLEW_INCLUDE_DIRS}
    ${tinyxml_INCLUDE_DIR}
)
link_directories(${tinyxml_LIBRARY_DIRS})
link_libraries(
    ${OPENGL_LIBRARIES}
    ${GLUT_glut_LIBRARY}
    ${GLEW_LIBRARIES}
    ${matheval_LIBRARIES}
    ${tinyxml_LIBRARIES}
)

# set dart source
set(dart_src
    ${PROJECT_SOURCE_DIR}/src/tracker.h
    ${PROJECT_SOURCE_DIR}/src/tracker.cpp

    ${PROJECT_SOURCE_DIR}/src/geometry/distance_transforms.h
    ${PROJECT_SOURCE_DIR}/src/geometry/geometry.h
    ${PROJECT_SOURCE_DIR}/src/geometry/geometry.cpp
    ${PROJECT_SOURCE_DIR}/src/geometry/grid_2d.h
    ${PROJECT_SOURCE_DIR}/src/geometry/grid_3d.h
    ${PROJECT_SOURCE_DIR}/src/geometry/sdf.h
    ${PROJECT_SOURCE_DIR}/src/geometry/sdf.cpp
    ${PROJECT_SOURCE_DIR}/src/geometry/SE3.h
    ${PROJECT_SOURCE_DIR}/src/geometry/plane_fitting.h
    ${PROJECT_SOURCE_DIR}/src/geometry/plane_fitting.cpp

    ${PROJECT_SOURCE_DIR}/src/mesh/mesh.h
    ${PROJECT_SOURCE_DIR}/src/mesh/mesh.cpp
    ${PROJECT_SOURCE_DIR}/src/mesh/mesh_proc.h
    ${PROJECT_SOURCE_DIR}/src/mesh/mesh_proc.cpp
    ${PROJECT_SOURCE_DIR}/src/mesh/mesh_splat.h
    ${PROJECT_SOURCE_DIR}/src/mesh/mesh_splat.cpp
    ${PROJECT_SOURCE_DIR}/src/mesh/mesh_sample.h
    ${PROJECT_SOURCE_DIR}/src/mesh/mesh_sample.cpp
    ${PROJECT_SOURCE_DIR}/src/mesh/primitive_meshing.h
    ${PROJECT_SOURCE_DIR}/src/mesh/primitive_meshing.cpp

    ${PROJECT_SOURCE_DIR}/src/img_proc/bilateral_filter.h
    ${PROJECT_SOURCE_DIR}/src/img_proc/img_ops.h
    ${PROJECT_SOURCE_DIR}/src/img_proc/organized_point_cloud.h
    ${PROJECT_SOURCE_DIR}/src/img_proc/resampling.h

    ${PROJECT_SOURCE_DIR}/src/model/host_only_model.h
    ${PROJECT_SOURCE_DIR}/src/model/host_only_model.cpp
    ${PROJECT_SOURCE_DIR}/src/model/mirrored_model.h
    ${PROJECT_SOURCE_DIR}/src/model/mirrored_model.cpp
    ${PROJECT_SOURCE_DIR}/src/model/model.h
    ${PROJECT_SOURCE_DIR}/src/model/model.cpp

    ${PROJECT_SOURCE_DIR}/src/optimization/kernels/obsToMod.h
    ${PROJECT_SOURCE_DIR}/src/optimization/kernels/modToObs.h
    ${PROJECT_SOURCE_DIR}/src/optimization/kernels/intersection.h
    ${PROJECT_SOURCE_DIR}/src/optimization/kernels/raycast.h
    ${PROJECT_SOURCE_DIR}/src/optimization/contact_prior.cpp
    ${PROJECT_SOURCE_DIR}/src/optimization/optimization.h
    ${PROJECT_SOURCE_DIR}/src/optimization/optimizer.h
    ${PROJECT_SOURCE_DIR}/src/optimization/optimizer.cpp
    ${PROJECT_SOURCE_DIR}/src/optimization/point_2d_3d_prior.cpp
    ${PROJECT_SOURCE_DIR}/src/optimization/point_3d_3d_prior.cpp
    ${PROJECT_SOURCE_DIR}/src/optimization/prediction_renderer.h
    ${PROJECT_SOURCE_DIR}/src/optimization/prediction_renderer.cpp
    ${PROJECT_SOURCE_DIR}/src/optimization/priors.h

    ${PROJECT_SOURCE_DIR}/src/pose/pose.h
    ${PROJECT_SOURCE_DIR}/src/pose/pose.cpp
    ${PROJECT_SOURCE_DIR}/src/pose/pose_reduction.h

    ${PROJECT_SOURCE_DIR}/src/util/cuda_utils.h
    ${PROJECT_SOURCE_DIR}/src/util/dart_io.h
    ${PROJECT_SOURCE_DIR}/src/util/dart_io.cpp
    ${PROJECT_SOURCE_DIR}/src/util/dart_types.h
    ${PROJECT_SOURCE_DIR}/src/util/gl_dart.h
    ${PROJECT_SOURCE_DIR}/src/util/gl_dart.cpp
    ${PROJECT_SOURCE_DIR}/src/util/mirrored_memory.h
    ${PROJECT_SOURCE_DIR}/src/util/model_renderer.h
    ${PROJECT_SOURCE_DIR}/src/util/model_renderer.cpp
    ${PROJECT_SOURCE_DIR}/src/util/ostream_operators.h
    ${PROJECT_SOURCE_DIR}/src/util/ostream_operators.cpp
    ${PROJECT_SOURCE_DIR}/src/util/prefix.h
    ${PROJECT_SOURCE_DIR}/src/util/string_format.h
    ${PROJECT_SOURCE_DIR}/src/util/string_format.cpp
    ${PROJECT_SOURCE_DIR}/src/util/vector_type_template.h
    ${PROJECT_SOURCE_DIR}/src/util/cpptoml.h

    ${PROJECT_SOURCE_DIR}/src/visualization/color_ramps.h
    ${PROJECT_SOURCE_DIR}/src/visualization/data_association_viz.h
    ${PROJECT_SOURCE_DIR}/src/visualization/gradient_viz.h
    ${PROJECT_SOURCE_DIR}/src/visualization/matrix_viz.h
    ${PROJECT_SOURCE_DIR}/src/visualization/sdf_viz.h
)

set(cpu_specific_src
    ${PROJECT_SOURCE_DIR}/src/geometry/distance_transforms.cpp
    ${PROJECT_SOURCE_DIR}/src/img_proc/img_ops.cpp
    ${PROJECT_SOURCE_DIR}/src/img_proc/organized_point_cloud.cpp
    ${PROJECT_SOURCE_DIR}/src/img_proc/resampling.cpp

    ${PROJECT_SOURCE_DIR}/src/visualization/color_ramps.cpp
    ${PROJECT_SOURCE_DIR}/src/visualization/data_association_viz.cpp
    ${PROJECT_SOURCE_DIR}/src/visualization/gradient_viz.cpp
    ${PROJECT_SOURCE_DIR}/src/visualization/sdf_viz.cpp
)

set(gpu_specific_src
    ${PROJECT_SOURCE_DIR}/src/geometry/distance_transforms.cu
    ${PROJECT_SOURCE_DIR}/src/geometry/plane_fitting.cu

    ${PROJECT_SOURCE_DIR}/src/img_proc/bilateral_filter.cu
    ${PROJECT_SOURCE_DIR}/src/img_proc/img_ops.cu
    ${PROJECT_SOURCE_DIR}/src/img_proc/organized_point_cloud.cu
    ${PROJECT_SOURCE_DIR}/src/img_proc/resampling.cu

    ${PROJECT_SOURCE_DIR}/src/optimization/kernels/kernel_common.h
    ${PROJECT_SOURCE_DIR}/src/optimization/kernels/obsToMod.cu
    ${PROJECT_SOURCE_DIR}/src/optimization/kernels/modToObs.cu
    ${PROJECT_SOURCE_DIR}/src/optimization/kernels/intersection.cu
    ${PROJECT_SOURCE_DIR}/src/optimization/kernels/raycast.cu

    ${PROJECT_SOURCE_DIR}/src/visualization/color_ramps.cu
    ${PROJECT_SOURCE_DIR}/src/visualization/data_association_viz.cu
    ${PROJECT_SOURCE_DIR}/src/visualization/gradient_viz.cu
    ${PROJECT_SOURCE_DIR}/src/visualization/matrix_viz.cu
    ${PROJECT_SOURCE_DIR}/src/visualization/sdf_viz.cu
)

set(point_cloud_srcs
    ${PROJECT_SOURCE_DIR}/src/point_cloud_src/point_cloud_src.h
)

if(assimp_FOUND)
    link_libraries(${assimp_LIBRARIES})
    include_directories(${assimp_INCLUDE_DIRS})
    set(dart_src
        ${dart_src}
        ${PROJECT_SOURCE_DIR}/src/mesh/assimp_mesh_reader.h
        ${PROJECT_SOURCE_DIR}/src/mesh/assimp_mesh_reader.cpp
    )
    add_definitions(-DASSIMP_BUILD)
endif()

include_directories(${PROJECT_SOURCE_DIR}/src/)

# set up library output
set(BUILD_SHARED_LIBS ON)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/lib)

set(CUDA_NVCC_FLAGS "-arch=sm_${CC}" "--use_fast_math"  "-O3" "--ptxas-options=--verbose") # "-fmad=false" "-DTHRUST_DEVICE_BACKEND=THRUST_DEVICE_BACKEND_OMP"

# add library
if (CUDA_FOUND)
    #include_directories(${CUDA_TOOLKIT_ROOT_DIR}/samples/common)

    add_custom_target(linkTarget COMMAND ln -sf ${CUDA_TOOLKIT_ROOT_DIR}/samples/common/inc/helper_math.h ${PROJECT_SOURCE_DIR}/src/util/helper_math.h)

    add_definitions(-DCUDA_BUILD)
    cuda_add_library(${PROJECT_NAME} ${dart_src} ${point_cloud_srcs} ${gpu_specific_src})
    add_dependencies(${PROJECT_NAME} linkTarget)

    message(STATUS "ln -s ${CUDA_TOOLKIT_ROOT_DIR}/samples/common/inc/helper_math.h ${PROJECT_SOURCE_DIR}/src/util/helper_math.h")


else()
    add_library(${PROJECT_NAME} ${dart_src} ${point_cloud_srcs} ${cpu_specific_src})
endif()

