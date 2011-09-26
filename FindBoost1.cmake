# Make sure that we can find Boost

# Try to automatically find boost
find_package(Boost REQUIRED)
if(Boost_FOUND)

  include_directories(${Boost_INCLUDE_DIR})

# If we can't, look in hard coded locations :-(
else()

  message(STATUS "Can't find Boost automatically, looking elsewhere...")

  find_path(
    Boost_INCLUDE_DIR boost
    /usr/local/boost
    /usr/local/boost_1_47_0
    /usr/local/boost_1_46_1
    /usr/local/boost_1_46_0
    /usr/local/boost_1_45_0
    /usr/local/boost_1_44_0
    /usr/local/boost_1_43_0
    /usr/local/boost_1_42_0
    /usr/local/boost_1_41_0
    /usr/local/boost_1_40_0
    /usr/local/include/boost
    /usr/local/include/boost_1_47_0
    /usr/local/include/boost_1_46_1
    /usr/local/include/boost_1_46_0
    /usr/local/include/boost_1_45_0
    /usr/local/include/boost_1_44_0
    /usr/local/include/boost_1_43_0
    /usr/local/include/boost_1_42_0
    /usr/local/include/boost_1_41_0
    /usr/local/include/boost_1_40_0
    /usr/include/boost
    /usr/include/boost_1_47_0
    /usr/include/boost_1_46_1
    /usr/include/boost_1_46_0
    /usr/include/boost_1_45_0
    /usr/include/boost_1_44_0
    /usr/include/boost_1_43_0
    /usr/include/boost_1_42_0
    /usr/include/boost_1_41_0
    /usr/include/boost_1_40_0
  )

  if(Boost_INCLUDE_DIR)
    include_directories(${Boost_INCLUDE_DIR})
    message(STATUS "Found Boost: ${Boost_INCLUDE_DIR}")
  else(Boost_INCLUDE_DIR)
    message(FATAL_ERROR "Boost not found")
  endif()

endif()
