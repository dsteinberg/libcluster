# Tools executable build instructions


#----------------------------------------------------#
# VDP cluster command line exec -- Work in progress  #
#----------------------------------------------------#

# Build against a pre-compiled library
add_executable(vdp_cluster ${TOOLS_SOURCE_DIR}/vdp_cluster.cpp)
target_link_libraries(vdp_cluster cluster)

# Install instructions
#install(TARGETS vdp_cluster DESTINATION bin) # Not actually using this anymore


#----------------------------------------#
# AUV pipeline clustering tools          #
#----------------------------------------#

if(BUILD_PIPELINE_TOOLS)

  # Seabed localisation
  set(LIBSEABEDCOMMON_INC ${AUVLIB_PREFIX}/include/libseabedcommon)
  set(LIBSEABEDCOMMON_LIBRARY seabedcommon)

  # libplankton
  set(LIBPLANKTON_INC ${AUVLIB_PREFIX}/include/libplankton)
  set(LIBPLANKTON_LIBRARY plankton gl2ps)

  # AUV data library
  set(LIBADT_INCLUDE_DIR ${AUVLIB_PREFIX}/include/libadt)
  set(LIBADT_LIBRARY adt)

  include_directories(
    ${LIBSEABEDCOMMON_INC}
    ${LIBPLANKTON_INC}
    ${LIBADT_INCLUDE_DIR}
  )

  add_executable(vdp_labels vdp_labels.cpp)

  target_link_libraries(vdp_labels
    cluster
    seabedcommon
    plankton
    adt
  )

  install(TARGETS vdp_labels DESTINATION bin)

endif(BUILD_PIPELINE_TOOLS)
