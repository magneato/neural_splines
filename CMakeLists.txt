cmake_minimum_required(VERSION 3.20)
project(NeuralSplines CXX)

# -----------------------------------------------------------------------------
# CMake configuration for the Neural Splines project
#
# This CMake file provides a reproducible build recipe for the native C++
# components of the project.  In particular it shows how to build an
# ExecuTorch runtime application that can execute a precompiled
# model (.pte) without the Python interpreter.  The C++ code lives in
# ``src/run_inference.cpp`` and demonstrates how to load a model,
# prepare inputs and execute the network.
#
# To configure and build the project run the following from a build
# directory:
#
#    mkdir build && cd build
#    cmake ..
#    make
#
# On success the ``run_inference`` executable will be generated in
# ``build``.  You can invoke it with the path to a .pte file and
# optionally a path to a CSV or binary file containing input data.
#
# Note that this build assumes the ExecuTorch C++ headers and
# libraries are installed in the active Python environment.  You can
# influence the search paths by setting ``EXECUTORCH_INCLUDE_DIR`` and
# ``EXECUTORCH_LIB_DIR`` on the CMake command line.  See the
# ``find_package`` call below for details.
# -----------------------------------------------------------------------------

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Attempt to locate the ExecuTorch headers and libraries by querying
# Python.  This method uses the active Python interpreter to locate
# the installed package.  Users may override these variables by
# specifying -DEXECUTORCH_INCLUDE_DIR=/path/to/include and
# -DEXECUTORCH_LIB_DIR=/path/to/lib when invoking CMake.
if(NOT EXECUTORCH_INCLUDE_DIR OR NOT EXECUTORCH_LIB_DIR)
    find_package(Python3 REQUIRED COMPONENTS Interpreter)
    execute_process(
        COMMAND ${Python3_EXECUTABLE} -c "import sysconfig, json, pathlib, importlib.util;\nimport executorch, sys;\nprint(executorch.__file__)"
        OUTPUT_VARIABLE _EXECUTORCH_INIT
        OUTPUT_STRIP_TRAILING_WHITESPACE
    )
    # The executorch module resides in .../site-packages/executorch/__init__.py
    # We derive include and lib directories relative to this location.
    get_filename_component(EXECUTORCH_PKG_DIR "${_EXECUTORCH_INIT}" DIRECTORY)
    if(NOT EXECUTORCH_INCLUDE_DIR)
        set(EXECUTORCH_INCLUDE_DIR "${EXECUTORCH_PKG_DIR}/include")
    endif()
    if(NOT EXECUTORCH_LIB_DIR)
        set(EXECUTORCH_LIB_DIR "${EXECUTORCH_PKG_DIR}/lib")
    endif()
endif()

message(STATUS "Using ExecuTorch include dir: ${EXECUTORCH_INCLUDE_DIR}")
message(STATUS "Using ExecuTorch library dir: ${EXECUTORCH_LIB_DIR}")

include_directories(${EXECUTORCH_INCLUDE_DIR})
link_directories(${EXECUTORCH_LIB_DIR})

add_executable(run_inference src/run_inference.cpp)

# Link against the ExecuTorch runtime.  On most platforms the library
# is named ``executorch``.  If you encounter unresolved symbols you
# may need to adjust this target or link additional static
# dependencies.  Consult the ExecuTorch build documentation for
# details.
target_link_libraries(run_inference PRIVATE executorch)