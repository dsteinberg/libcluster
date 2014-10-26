if(BUILD_PYTHON_INTERFACE)

  message(STATUS "Will build the python interface")
  if(BUILD_USE_PYTHON3)
      message(STATUS "Will use python 3")
  else(BUILD_USE_PYTHON3)
      message(STATUS "Will use python 2")
  endif(BUILD_USE_PYTHON3)

  # Python needs row major matrices (for convenience)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DEIGEN_DEFAULT_TO_ROW_MAJOR")


  #--------------------------------#
  # Includes                       #
  #--------------------------------#

  if(BUILD_USE_PYTHON3)
    find_package(Boost COMPONENTS python3 REQUIRED)
  else(BUILD_USE_PYTHON3)
    find_package(Boost COMPONENTS python REQUIRED)
  endif(BUILD_USE_PYTHON3)
  find_package(PythonLibs REQUIRED)
  include_directories(${PYTHON_INCLUDE_DIRS})


  #--------------------------------#
  # Library Build Instructions     #
  #--------------------------------#

  add_library(${PROJECT_NAME}py SHARED
    ${PYTHON_SOURCE_DIR}/libclusterpy.h
    ${PYTHON_SOURCE_DIR}/libclusterpy.cpp
  )

  if(BUILD_USE_PYTHON3)
    set(BOOST_PYTHON boost_python3)
  else(BUILD_USE_PYTHON3)
    set(BOOST_PYTHON boost_python)
  endif(BUILD_USE_PYTHON3)

  target_link_libraries(${PROJECT_NAME}py
      ${BOOST_PYTHON}
      ${PYTHON_LIBRARIES}
      ${Boost_LIBRARIES}
      ${PROJECT_NAME}
  )


  #--------------------------------#
  # Install Instructions           #
  #--------------------------------#

  if(BUILD_USE_PYTHON3)
      set(PYCMD "python3")
  else(BUILD_USE_PYTHON3)
      set(PYCMD "python2")
  endif(BUILD_USE_PYTHON3)

  # Get python path
  execute_process(COMMAND ${PYCMD} -c
    "from distutils.sysconfig import get_python_lib; print(get_python_lib())"
    OUTPUT_VARIABLE PYTHON_SITE_PACKAGES OUTPUT_STRIP_TRAILING_WHITESPACE
  )

  # Install target
  install(TARGETS ${PROJECT_NAME}py DESTINATION ${PYTHON_SITE_PACKAGES})

endif(BUILD_PYTHON_INTERFACE)
