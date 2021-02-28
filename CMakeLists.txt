cmake_minimum_required(VERSION 3.14 FATAL_ERROR)
SET(CMAKE_C_COMPILER /usr/bin/clang)
SET(CMAKE_CXX_COMPILER /usr/bin/clang++)
#SET(CMAKE_CXX_FLAGS "-Ofast") # Optimizing
SET(CMAKE_CXX_FLAGS "-g -O0") # Debugging
project(main)

find_package(OpenCV REQUIRED)
find_package(fmt REQUIRED)

### dlib
include(FetchContent)
FetchContent_Declare(
    dlib
    GIT_REPOSITORY https://github.com/davisking/dlib.git
    GIT_TAG v19.21
    )
FetchContent_MakeAvailable(dlib)


add_executable(main main.cpp detector/detector.cpp recognizor/recognizor.cpp util/performance.cpp)

target_link_libraries(main ${OpenCV_LIBS} dlib::dlib fmt::fmt)

target_include_directories(main PRIVATE detector recognizor util ${OpenCV_INCLUDE_DIRS})

set_property(TARGET main PROPERTY CXX_STANDARD 20)
set_property(TARGET main PROPERTY RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/..")

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_VERBOSE_MAKEFILE ON)
