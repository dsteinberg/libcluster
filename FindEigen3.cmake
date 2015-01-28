# Make sure that we can find Eigen
# This creates the following variables:
#  - EIGEN_INCLUDE_DIRS where to find the library
#  - EIGEN_FOUND TRUE if found, FALSE otherwise

find_path(
  EIGEN_INCLUDE_DIRS Eigen
  /usr/local/eigen3
  /usr/local/include/eigen3
  /usr/include/eigen3
)

# Check found Eigen
if(EIGEN_INCLUDE_DIRS)
  set(EIGEN_FOUND TRUE)
  message(STATUS "Found Eigen: ${EIGEN_INCLUDE_DIRS}")
else(EIGEN_INCLUDE_DIRS)
  if(EIGEN_FIND_REQUIRED)
    set(EIGEN_FOUND FALSE)
    message(FATAL_ERROR "Eigen not found")
  endif(EIGEN_FIND_REQUIRED)
endif(EIGEN_INCLUDE_DIRS)
