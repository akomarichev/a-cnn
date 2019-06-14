cmake_minimum_required(VERSION 2.6 FATAL_ERROR)

project(MY_GRAND_PROJECT)

find_package(PCL 1.3 REQUIRED COMPONENTS common io)

list(REMOVE_ITEM PCL_LIBRARIES "vtkproj4")

include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

add_executable(2_compute_normals_and_curvature 2_compute_normals_and_curvature.cpp)
target_link_libraries(2_compute_normals_and_curvature ${PCL_COMMON_LIBRARIES} ${PCL_IO_LIBRARIES} pcl_common pcl_kdtree pcl_search pcl_features pcl_io)
