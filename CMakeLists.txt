project(cluster)
cmake_minimum_required(VERSION 2.6)

#--------------------------------#
# Includes                       #
#--------------------------------#

include(${PROJECT_SOURCE_DIR}/FindBoost1.cmake REQUIRED)
include(${PROJECT_SOURCE_DIR}/../CMakeUtils/FindEigen3.cmake REQUIRED)

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

# Locations for source code
set(LIB_SOURCE_DIR   ${PROJECT_SOURCE_DIR}/src)
set(LIB_INCLUDE_DIR  ${PROJECT_SOURCE_DIR}/include)
set(TEST_SOURCE_DIR  ${PROJECT_SOURCE_DIR}/test)
set(TOOLS_SOURCE_DIR ${PROJECT_SOURCE_DIR}/tools)

# Locations for binary files
set(LIBRARY_OUTPUT_PATH     ${PROJECT_SOURCE_DIR}/lib)
set(EXECUTABLE_OUTPUT_PATH  ${PROJECT_SOURCE_DIR}/build)

# Automatically or from command line set build type
if(NOT CMAKE_BUILD_TYPE)
  set( CMAKE_BUILD_TYPE Release CACHE STRING
       "Build type options are: None Debug Release RelWithDebInfo MinSizeRel."
       FORCE
     )
endif(NOT CMAKE_BUILD_TYPE)

# Path to AUV libs
if(NOT AUVLIB_PREFIX)
   set(AUVLIB_PREFIX /usr/local/)
endif(NOT AUVLIB_PREFIX)

# Some options
option(BUILD_PIPELINE_TOOLS "Build clustering tools for the AUV pipeline" on)

#--------------------------------#
# Library Build Instructions     #
#--------------------------------#

# Make sure we include library headers in compile
include_directories(${LIB_INCLUDE_DIR})

# Library build instructions
add_library(cluster SHARED
  ${LIB_INCLUDE_DIR}/libcluster.h
  ${LIB_INCLUDE_DIR}/probutils.h
  ${LIB_SOURCE_DIR}/vbcommon.h
  ${LIB_SOURCE_DIR}/vbcommon.cpp
  ${LIB_SOURCE_DIR}/gmm.cpp
  ${LIB_SOURCE_DIR}/vdp.cpp
  ${LIB_SOURCE_DIR}/gmc.cpp
  ${LIB_SOURCE_DIR}/igmc.cpp
  ${LIB_SOURCE_DIR}/probutils.cpp
)

add_definitions("-Wall")

#--------------------------------#
# Library Install Instructions   #
#--------------------------------#

if(NOT CMAKE_INSTALL_PREFIX )
   set(CMAKE_INSTALL_PREFIX "/usr/local" )
endif(NOT CMAKE_INSTALL_PREFIX )

install(TARGETS cluster DESTINATION lib)
install(FILES
  ${LIB_INCLUDE_DIR}/libcluster.h
  ${LIB_INCLUDE_DIR}/probutils.h
  DESTINATION include/libcluster
)

#--------------------------------#
# Subdirectories to recurse to   #
#--------------------------------#

subdirs(test tools)
