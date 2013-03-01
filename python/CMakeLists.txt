if(BUILD_PYTHON_INTERFACE)

  message(STATUS "Will build the python interface")

  add_library(${PROJECT_NAME}py SHARED
    ${PYTHON_SOURCE_DIR}/pintfctns.h
    ${PYTHON_SOURCE_DIR}/pintfctns.cpp
  )

endif(BUILD_PYTHON_INTERFACE)
