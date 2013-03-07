project(cluster)
cmake_minimum_required(VERSION 2.6)

#--------------------------------#
# Includes                       #
#--------------------------------#

#include(${PROJECT_SOURCE_DIR}/FindBoost1.cmake REQUIRED)
find_package(Boost REQUIRED)
include_directories(${Boost_INCLUDE_DIRS})
include(${PROJECT_SOURCE_DIR}/FindEigen3.cmake REQUIRED)
include(FindOpenMP)

#--------------------------------#
# Enforce an out-of-source build #
#--------------------------------#

string(COMPARE EQUAL "${PROJECT_SOURCE_DIR}" "${PROJECT_BINARY_DIR}" INSOURCE)
if(INSOURCE)
  message(FATAL_ERROR "This project requires an out of source build. This is handled by the Makefile. Please type 'make'.")
endif(INSOURCE)

#--------------------------------#
# Compiler environment Setup     #
#--------------------------------#

# Some compilation options (changeable from ccmake)
option(BUILD_EXHAUST_SPLIT "Use the exhaustive cluster split heuristic?" off)
option(BUILD_PYTHON_INTERFACE "Build the python interface?" off)

# Locations for source code
set(LIB_SOURCE_DIR    ${PROJECT_SOURCE_DIR}/src)
set(LIB_INCLUDE_DIR   ${PROJECT_SOURCE_DIR}/include)
set(TEST_SOURCE_DIR   ${PROJECT_SOURCE_DIR}/test)
set(MATLAB_SOURCE_DIR ${PROJECT_SOURCE_DIR}/matlab)
set(PYTHON_SOURCE_DIR ${PROJECT_SOURCE_DIR}/python)

# Locations for binary files
set(LIBRARY_OUTPUT_PATH     ${PROJECT_SOURCE_DIR}/lib)
set(EXECUTABLE_OUTPUT_PATH  ${PROJECT_SOURCE_DIR}/build)

# Automatically or from command line set build type
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING
       "Build type options are: None Debug Release RelWithDebInfo MinSizeRel."
       FORCE
     )
endif(NOT CMAKE_BUILD_TYPE)

# If we want to use the greedy splitting heuristic, define it here
if(BUILD_EXHAUST_SPLIT)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DEXHAUST_SPLIT")
endif(BUILD_EXHAUST_SPLIT)

if(BUILD_PYTHON_INTERFACE)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DEIGEN_DEFAULT_TO_ROW_MAJOR")
endif(BUILD_PYTHON_INTERFACE)

# Search for OpenMP support for multi-threading
if(OPENMP_FOUND)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
  set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif(OPENMP_FOUND)

#--------------------------------#
# Library Build Instructions     #
#--------------------------------#

# Make sure we include library headers in compile
include_directories(${LIB_INCLUDE_DIR})

# Library build instructions
add_library(${PROJECT_NAME} SHARED
  ${LIB_INCLUDE_DIR}/libcluster.h
  ${LIB_INCLUDE_DIR}/probutils.h
  ${LIB_INCLUDE_DIR}/distributions.h
  ${LIB_SOURCE_DIR}/distributions.cpp
  ${LIB_SOURCE_DIR}/comutils.h
  ${LIB_SOURCE_DIR}/comutils.cpp
  ${LIB_SOURCE_DIR}/cluster.cpp
  ${LIB_SOURCE_DIR}/scluster.cpp
  ${LIB_SOURCE_DIR}/mcluster.cpp
  ${LIB_SOURCE_DIR}/probutils.cpp
)

add_definitions("-Wall")

#--------------------------------#
# Library Install Instructions   #
#--------------------------------#

if(NOT CMAKE_INSTALL_PREFIX)
   set(CMAKE_INSTALL_PREFIX "/usr/local" )
endif(NOT CMAKE_INSTALL_PREFIX)

install(TARGETS ${PROJECT_NAME} DESTINATION lib)
install(FILES
  ${LIB_INCLUDE_DIR}/libcluster.h
  ${LIB_INCLUDE_DIR}/probutils.h
  ${LIB_INCLUDE_DIR}/distributions.h
  DESTINATION include/libcluster
)

#--------------------------------#
# Subdirectories to recurse to   #
#--------------------------------#

subdirs(test matlab python)
