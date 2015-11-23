cmake_minimum_required(VERSION 2.4.6)
include($ENV{ROS_ROOT}/core/rosbuild/rosbuild.cmake)

# Set the build type.  Options are:
#  Coverage       : w/ debug symbols, w/o optimization, w/ code-coverage
#  Debug          : w/ debug symbols, w/o optimization
#  Release        : w/o debug symbols, w/ optimization
#  RelWithDebInfo : w/ debug symbols, w/ optimization
#  MinSizeRel     : w/o debug symbols, w/ optimization, stripped binaries
#set(ROS_BUILD_TYPE RelWithDebInfo)

rosbuild_init()

set(CMAKE_CXX_FLAGS "-fPIC")

PROJECT(ISL_localization)
FIND_PACKAGE(Qt4 REQUIRED)
SET(QT_USE_QTSQL TRUE)

FIND_PACKAGE( OpenCV REQUIRED )

INCLUDE(${QT_USE_FILE})
ADD_DEFINITIONS(${QT_DEFINITIONS})

SET(ISL_localization_SOURCES src/bubble_space_node.cpp src/bubbleprocess/bubbleprocess.cpp src/imageprocess/imageprocess.cpp src/bubbleprocess/dfc.cpp src/bubbleprocess/invariants.cpp src/database/databasemanager.cpp src/localization/localization.cpp)
SET(ISL_localization_HEADERS src/bubbleprocess/bubbleprocess.h src/imageprocess/imageprocess.h src/database/databasemanager.h src/localization/localization.h)

QT4_WRAP_CPP(ISL_localization_HEADERS_MOC ${ISL_localization_HEADERS})
qt4_automoc(${ISL_localization_SOURCES})


#set the default path for built executables to the "bin" directory
set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin)
#set the default path for built libraries to the "lib" directory
set(LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/lib)

#uncomment if you have defined messages
#rosbuild_genmsg()
#uncomment if you have defined services
#rosbuild_gensrv()

#common commands for building c++ executables and libraries
#rosbuild_add_library(${PROJECT_NAME} src/example.cpp)
#target_link_libraries(${PROJECT_NAME} another_library)
#rosbuild_add_boost_directories()
#rosbuild_link_boost(${PROJECT_NAME} thread)
#rosbuild_add_executable(example examples/example.cpp)
#target_link_libraries(example ${PROJECT_NAME})

include_directories(${CMAKE_CURRENT_BINARY_DIR})
include_directories(/opt/ros/fuerte/stacks)
include_directories(./src/pcprocessing)
include_directories(./src/bubbleprocess)
include_directories(./src/imageprocess)
include_directories(./Qt-Build)
include_directories(./src/database)
include_directories(./src/localization)

rosbuild_add_library(${PROJECT_NAME} src/bubble_space_node.cpp)

target_link_libraries(${PROJECT_NAME} opencv_highgui opencv_imgproc opencv_core ${QT_LIBRARIES})

rosbuild_add_executable(bubble_space_node ${ISL_localization_SOURCES} ${ISL_localization_HEADERS_MOC})

target_link_libraries(bubble_space_node ${PROJECT_NAME})
