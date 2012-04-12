# Make sure that we can find Eigen
find_path(
  EIGEN_INCLUDE_DIRS Eigen
  /usr/local/eigen3
  /usr/local/include/eigen3
  /usr/include/eigen3
)

# Include Eigen
if(EIGEN_INCLUDE_DIRS)
  include_directories(${EIGEN_INCLUDE_DIRS})
  message(STATUS "Found Eigen: ${EIGEN_INCLUDE_DIRS}")
else(EIGEN_INCLUDE_DIRS)
  message(FATAL_ERROR "Eigen not found")
endif(EIGEN_INCLUDE_DIRS)
