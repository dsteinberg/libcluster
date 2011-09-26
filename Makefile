#
# AUV build system. The script common.mk calls cmake to create an out-of-source
# build. Temporary files are kept in the 'build' directory.
#
# Compilation:
#    'make'          compile (a release build will be created by default)
#    'make install'  compile and install
#
# Build types:
#    'make debug'           not optimised, debugging symbols
#    'make release'         optimised for speed, no debugging symbols 
#    'make relwithdebinfo'  optimised for speed, debugging symbols
#    'make minsizerel'      optimised for size, no debugging symbols
#
# Clean-up:
#    'make clean'      remove local files built by the 'make' command
#    'make uninstall'  remove files installed by the 'make install' command
#
# Documentation
#    'make doc' to build Doxygen documentation
#
# Testing
#    'make test' to run tests
#
include ${CURDIR}/../CMakeUtils/common.mk
