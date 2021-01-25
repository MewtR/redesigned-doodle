cmake_minimum_required(VERSION 3.14 FATAL_ERROR)
project(main)

find_package(OpenCV REQUIRED)
include_directories( ${OpenCV_INCLUDE_DIRS})

### dlib
include(FetchContent)
FetchContent_Declare(
    dlib
    GIT_REPOSITORY https://github.com/davisking/dlib.git
    GIT_TAG v19.21
    )
FetchContent_MakeAvailable(dlib)


add_executable(main main.cpp)
target_link_libraries(main ${OpenCV_LIBS} dlib::dlib)
set_property(TARGET main PROPERTY CXX_STANDARD 20)
set_property(TARGET main PROPERTY RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/..")
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
