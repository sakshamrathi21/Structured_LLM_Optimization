cmake_minimum_required(VERSION 3.14)
project(chunked_hash_tree)

find_package(pybind11 REQUIRED)

add_library(chunked_hash_tree MODULE
    chunked_hash_tree.cpp
    chunked_hash_tree_rl.cpp
    bindings.cpp
)

target_link_libraries(chunked_hash_tree PRIVATE pybind11::module xxhash)
set_target_properties(chunked_hash_tree PROPERTIES PREFIX "")
