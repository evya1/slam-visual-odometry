cmake_minimum_required(VERSION 3.10)
project(slam_visual_odometry VERSION 0.1.0 LANGUAGES CXX)

# Set C++17 standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Set output directories
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

# Include directories
include_directories(
    ${PROJECT_SOURCE_DIR}/include
)

# Options
option(BUILD_TESTS "Build tests" ON)

# Find dependencies (placeholder for future dependencies)
# find_package(OpenCV REQUIRED)
# find_package(Eigen3 REQUIRED)
# find_package(Pangolin REQUIRED)

# Add subdirectories (when source files are added)
# add_subdirectory(src)

# Testing
if(BUILD_TESTS)
    enable_testing()
    # add_subdirectory(tests)
endif()

# Print configuration summary
message(STATUS "")
message(STATUS "Configuration Summary:")
message(STATUS "  Project: ${PROJECT_NAME} ${PROJECT_VERSION}")
message(STATUS "  C++ Standard: ${CMAKE_CXX_STANDARD}")
message(STATUS "  Build Type: ${CMAKE_BUILD_TYPE}")
message(STATUS "  Build Tests: ${BUILD_TESTS}")
message(STATUS "")
