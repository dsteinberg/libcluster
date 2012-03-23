# Test executable build instructions

# Make GMM batch test executable (test VDP and GMC)
add_executable(gmm_test
  ${TEST_SOURCE_DIR}/gmm_test.cpp
  ${TEST_SOURCE_DIR}/testdata.h
)

target_link_libraries(gmm_test cluster)
