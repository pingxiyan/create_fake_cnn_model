CMAKE_MINIMUM_REQUIRED(VERSION 3.2)

SET (TARGET_NAME caffe_test)
PROJECT(${TARGET_NAME})

MESSAGE(STATUS "==================================-")
MESSAGE(STATUS "Start:${TARGET_NAME}")

FIND_PACKAGE( OpenCV REQUIRED )
FIND_PACKAGE( Caffe REQUIRED HINTS /home/xiping/mygithub/create_fake_cnn_model/fake_caffe/caffe/build_cpu/install)

ADD_DEFINITIONS(-std=c++11)
#==========================================
# Set include path
INCLUDE_DIRECTORIES(
	${OpenCV_INCLUDE_DIRS}
	${CMAKE_CURRENT_SOURCE_DIR}
	)

ADD_EXECUTABLE(${TARGET_NAME} 
    ${CMAKE_CURRENT_SOURCE_DIR}/main.cpp 
	${CMAKE_CURRENT_SOURCE_DIR}/region_yolov2tiny.cpp)


TARGET_LINK_LIBRARIES(${TARGET_NAME} ${OpenCV_LIBS} 
	${Caffe_LIBRARIES}
	)
