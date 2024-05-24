cmake_minimum_required(VERSION 3.2)

project(Krcma_Ukol_2 LANGUAGES CUDA CXX)

find_package(PNG REQUIRED)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(Krcma_Ukol_2 main.cu headers.h opt_function.h cpu_generator.h gpu_generator.cu cpu_image.h gpu_image.cu utils/pngio.cpp)

target_link_libraries(Krcma_Ukol_2 ${PNG_LIBRARY})