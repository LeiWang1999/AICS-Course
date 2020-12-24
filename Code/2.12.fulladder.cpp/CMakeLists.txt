cmake_minimum_required(VERSION 3.16.5)
project(2_12_fulladder_cpp)

set(CMAKE_CXX_STANDARD 14)
add_executable(2_12_fulladder_cpp main.cpp data/binary_adder.cpp data/binary_adder.h data/data_gen.cpp data/data_gen.h Net/MLP.cpp Net/MLP.h)
#set(CMAKE_CXX_FLAGS "-lstdc++")
