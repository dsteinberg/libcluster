# Find the Matlab package -- not necessary, just for IDE niceties
find_package(Matlab)

if(MATLAB_INCLUDE_DIR)
  include_directories(${MATLAB_INCLUDE_DIR})
  message(STATUS "Found Matlab: ${MATLAB_INCLUDE_DIR}")
endif(MATLAB_INCLUDE_DIR)

#--------------------------------#
# List of mex files to build     #
#--------------------------------#

# Make a list of files to compile -- All files in this list will be compiled
set(COMPILE_LIST
    learnVDP.cpp
    learnBGMM.cpp
    learnGMC.cpp
    learnSCM.cpp
)

#--------------------------------#
# Mex compile settings           #
#--------------------------------#

# C++ FLAGS for compiling the mex files
set(MAT_FLAGS
    ${CMAKE_CXX_FLAGS}
    ${CMAKE_EXE_LINKER_FLAGS}
)

# Includes required for the mex files
set(MAT_INCS
    -I${EIGEN_INCLUDE_DIRS}
    -I${LIB_INCLUDE_DIR}
)

# Location of static objects required for linking to the mex files
set(MAT_LINKS
      $<TARGET_LINKER_FILE:${PROJECT_NAME}>
      intfctns.cpp
)

#--------------------------------#
# Mex compile instructions       #
#--------------------------------#

# Make the Rules/commands for compiling the cpp files in COMPILE_LIST
foreach(mexfile ${COMPILE_LIST})
  add_custom_command(
    OUTPUT ${mexfile}
    COMMAND mex ${MAT_FLAGS} ${MAT_INCS} ${mexfile} ${MAT_LINKS}
    WORKING_DIRECTORY ${MATLAB_SOURCE_DIR}
  )
endforeach(mexfile)

# Custom target for builing the Matlab interface - calls all custom commands
add_custom_target(matint
  DEPENDS ${PROJECT_NAME} ${COMPILE_LIST}
  SOURCES intfctns.cpp ${COMPILE_LIST}
)

#--------------------------------#
# Mex clean instructions, TODO   #
#--------------------------------#

#set_directory_properties(PROPERTIES
#  ADDITIONAL_MAKE_CLEAN_FILES "${MATLAB_SOURCE_DIR}/learnVDP.mex*"
#)
