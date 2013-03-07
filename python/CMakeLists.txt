if(BUILD_PYTHON_INTERFACE)

  message(STATUS "Will build the python interface")

  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DEIGEN_DEFAULT_TO_ROW_MAJOR")

  #--------------------------------#
  # Includes                       #
  #--------------------------------#

  find_package(Boost COMPONENTS python REQUIRED)
  find_package(PythonLibs REQUIRED)
  include_directories(${PYTHON_INCLUDE_DIRS})


  #--------------------------------#
  # Library Build Instructions     #
  #--------------------------------#

  add_library(${PROJECT_NAME}py SHARED
    ${PYTHON_SOURCE_DIR}/libclusterpy.h
    ${PYTHON_SOURCE_DIR}/libclusterpy.cpp
  )

  target_link_libraries(${PROJECT_NAME}py
      boost_python
      ${PYTHON_LIBRARIES}
      ${Boost_LIBRARIES}
      ${PROJECT_NAME}
  )

endif(BUILD_PYTHON_INTERFACE)
