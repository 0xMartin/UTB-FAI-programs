cmake_minimum_required(VERSION 3.5)

project(krcma_ukol_5 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(krcma_ukol_5 main.cpp)

install(TARGETS krcma_ukol_5
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR})

find_package(OpenMP REQUIRED)

if(OpenMP_CXX_FOUND)
    target_compile_options(krcma_ukol_5 PUBLIC ${OpenMP_CXX_FLAGS})
    target_link_libraries(krcma_ukol_5 PUBLIC OpenMP::OpenMP_CXX)
endif()

find_package(Boost REQUIRED COMPONENTS program_options REQUIRED)
include_directories(${Boost_INCLUDE_DIR})
target_link_libraries(krcma_ukol_5 PUBLIC ${Boost_LIBRARIES})
