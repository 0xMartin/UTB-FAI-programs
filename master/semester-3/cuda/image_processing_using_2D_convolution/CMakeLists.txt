cmake_minimum_required(VERSION 3.2)

project(Krcma_Ukol_3 LANGUAGES CUDA CXX)

find_package(PNG REQUIRED)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(Krcma_Ukol_3 main.cu convolution.cu image_rgb.cu mask_utils.cpp argument_parser.cpp utils/pngio.cpp)

target_link_libraries(Krcma_Ukol_3 ${PNG_LIBRARY})
