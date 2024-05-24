cmake_minimum_required(VERSION 3.2)

project(Krcma_Ukol_4 LANGUAGES CUDA CXX)

find_package(PNG REQUIRED)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(Krcma_Ukol_4 main.cu image_rgb.cu histogram_generator.cu argument_parser.cpp utils/pngio.cpp)

target_link_libraries(Krcma_Ukol_4 ${PNG_LIBRARY})
