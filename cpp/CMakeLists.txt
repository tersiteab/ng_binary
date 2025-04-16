cmake_minimum_required(VERSION 3.13)
project(pointpillars_driver)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Adjust this to your actual TVM path
set(TVM_HOME "/home/tersiteab/apache-tvm-src-v0.18.0")
set(TVM_RUNTIME_LIB "${TVM_HOME}/build")
set(PILLAR_HOME "/home/tersiteab/Documents/PointCloudResearch/PointPillars_HSC/src")
# Include directories
include_directories(
    ${TVM_HOME}/include
    ${TVM_HOME}/3rdparty/dmlc-core/include
    ${TVM_HOME}/3rdparty/dlpack/include
)

# Source file
add_executable(pointpillars_driver main.cpp)  # or your actual file name

# Link the TVM runtime and CUDA
target_link_libraries(pointpillars_driver
    ${TVM_RUNTIME_LIB}/libtvm_runtime.so
    # ${TVM_RUNTIME_LIB}/libtvm.so
    pthread
    dl
    stdc++fs
    ${PILLAR_HOME}/libpillars.so
    )


# Set rpath so you don't have to export LD_LIBRARY_PATH at runtime
# set_target_properties(pointpillars_driver PROPERTIES
#     BUILD_RPATH ${TVM_RUNTIME_LIB}
# )
set_target_properties(pointpillars_driver PROPERTIES
    BUILD_RPATH "${TVM_RUNTIME_LIB};${PILLAR_HOME}"
)
