# Test executable build instructions

# Make Cluster models batch test executable (test VDP and GMC)
add_executable(cluster_test
  ${TEST_SOURCE_DIR}/cluster_test.cpp
  ${TEST_SOURCE_DIR}/testdata.h
)

target_link_libraries(cluster_test cluster)

# Make Topic models batch test executable
add_executable(topic_test
  ${TEST_SOURCE_DIR}/topic_test.cpp
  ${TEST_SOURCE_DIR}/testdata.h
)

target_link_libraries(topic_test cluster)
