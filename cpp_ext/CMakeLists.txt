cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(memory_ext)

find_package(Torch REQUIRED)

add_library(memory_ext SHARED memory_ext.cpp)
target_link_libraries(memory_ext "${TORCH_LIBRARIES}")
set_property(TARGET memory_ext PROPERTY CXX_STANDARD 14)

# Nouveau : définir l'emplacement de sortie directement
set_target_properties(memory_ext PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/../python
)

