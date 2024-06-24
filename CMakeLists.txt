cmake_minimum_required(VERSION 3.0)

project(TPE)

add_executable(TPE main.cpp)

add_library(tpe SHARED tokenizer.cpp)