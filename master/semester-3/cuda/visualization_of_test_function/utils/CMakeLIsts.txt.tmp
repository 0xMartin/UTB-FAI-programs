cmake_minimum_required(VERSION 3.0)

project(Lesson3 LANGUAGES CXX CUDA) 

find_package(PNG REQUIRED)
find_package(OpenMP REQUIRED)

list(APPEND CMAKE_CXX_FLAGS "${OpenMP_CXX_FLAGS}")
add_executable(Lesson3 main.cu pngio.cpp)

target_link_libraries(Lesson3 ${PNG_LIBRARY})