CMAKE_MINIMUM_REQUIRED(VERSION 3.2)

SET (TARGET_NAME caffe_test)
PROJECT(${TARGET_NAME})

MESSAGE(STATUS "==================================-")
MESSAGE(STATUS "Start:${TARGET_NAME}")

FIND_PACKAGE( OpenCV REQUIRED )
# find myown build caffe
SET(CAFFE_PATH /home/xiping/mygithub/chenwei803/person_analysis/call_caffe)
include(${CAFFE_PATH}/caffe_src_code/caffe.cmake)

ADD_DEFINITIONS(-std=c++11)
#==========================================
# Set include path
INCLUDE_DIRECTORIES(
	${OpenCV_INCLUDE_DIRS}
	${CMAKE_CURRENT_SOURCE_DIR}
	)

ADD_EXECUTABLE(${TARGET_NAME} 
    ${CMAKE_CURRENT_SOURCE_DIR}/main.cpp)


TARGET_LINK_LIBRARIES(${TARGET_NAME} ${OpenCV_LIBS} 
	${caffe_libs} 
	)
