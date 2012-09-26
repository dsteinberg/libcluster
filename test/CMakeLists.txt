# Test executable build instructions

# Make Cluster models batch test executable (test VDP and GMC)
add_executable(cluster_test
  ${TEST_SOURCE_DIR}/cluster_test.cpp
  ${TEST_SOURCE_DIR}/testdata.h
)

target_link_libraries(cluster_test ${PROJECT_NAME})

# Make Topic models batch test executable
add_executable(scluster_test
  ${TEST_SOURCE_DIR}/scluster_test.cpp
  ${TEST_SOURCE_DIR}/testdata.h
)

target_link_libraries(scluster_test ${PROJECT_NAME})
