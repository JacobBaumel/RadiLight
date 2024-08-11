cmake_minimum_required(VERSION 3.15)
project(opencvTest)

find_package(OpenCV REQUIRED)
find_package(CUDA REQUIRED)

include_directories(${CUDA_INCLUDE_DIRS})
include_directories(${OpenCV_INCLUDE_DIRS})

cuda_add_executable(opencvTest main.cu)

add_library(libapril_tagging STATIC IMPORTED)
set_target_properties(libapril_tagging PROPERTIES IMPORTED_LOCATION /home/nano/RadiLight/nvapriltags/libapril_tagging.a)
set_target_properties(libapril_tagging PROPERTIES INTERFACE_INCLUDE_DIRECTORIES /home/nano/RadiLight/nvapriltags/include)
target_link_libraries(opencvTest libapril_tagging)

target_link_libraries(opencvTest ${CUDA_LIBRARIES} ${OpenCV_LIBS})
