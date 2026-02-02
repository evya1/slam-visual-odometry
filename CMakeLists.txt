cmake_minimum_required(VERSION 3.16)

project(slam_visual_odometry VERSION 0.1.0 LANGUAGES CXX)

# ---------- Global defaults ----------
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE RelWithDebInfo CACHE STRING "Build type" FORCE)
endif()

# ---------- Options ----------
option(SLAM_VO_BUILD_TESTS "Build tests" ON)
option(SLAM_VO_REQUIRE_DEPS "Fail configure if OpenCV/Eigen/Pangolin not found" OFF)

# ---------- Project-wide interface target ----------
add_library(slam_vo_options INTERFACE)

target_compile_features(slam_vo_options INTERFACE cxx_std_17)

target_include_directories(slam_vo_options INTERFACE
  $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)

# ---------- Dependencies ----------
# Keep placeholder-safe: only require deps if you explicitly turn it on.
if(SLAM_VO_REQUIRE_DEPS)
  find_package(OpenCV REQUIRED)
  find_package(Eigen3 REQUIRED)
  find_package(Pangolin REQUIRED)
endif()

# ---------- Add subdirectories only if they exist ----------
if(EXISTS "${PROJECT_SOURCE_DIR}/src/CMakeLists.txt")
  add_subdirectory(src)
endif()

if(SLAM_VO_BUILD_TESTS)
  enable_testing()
  if(EXISTS "${PROJECT_SOURCE_DIR}/tests/CMakeLists.txt")
    add_subdirectory(tests)
  endif()
endif()

# ---------- Summary ----------
message(STATUS "")
message(STATUS "Configuration Summary:")
message(STATUS "  Project:       ${PROJECT_NAME} ${PROJECT_VERSION}")
message(STATUS "  C++ Standard:  ${CMAKE_CXX_STANDARD}")
message(STATUS "  Build Type:    ${CMAKE_BUILD_TYPE}")
message(STATUS "  Build Tests:   ${SLAM_VO_BUILD_TESTS}")
message(STATUS "  Require Deps:  ${SLAM_VO_REQUIRE_DEPS}")
message(STATUS "")
