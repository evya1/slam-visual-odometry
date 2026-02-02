cmake_minimum_required(VERSION 3.16)

project(slam_visual_odometry VERSION 0.1.0 LANGUAGES CXX)

# ---- Defaults ----
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Default build type for single-config generators (Ninja/Make)
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE RelWithDebInfo CACHE STRING "Build type" FORCE)
endif()

# ---- Options ----
option(SLAM_VO_BUILD_TESTS "Build tests" ON)
option(SLAM_VO_REQUIRE_DEPS "Fail configure if OpenCV/Eigen/Pangolin not found" OFF)
option(SLAM_VO_ENABLE_WARNINGS "Enable common compiler warnings" ON)

# ---- One project-wide interface target (everything links to this later) ----
add_library(slam_vo INTERFACE)
target_compile_features(slam_vo INTERFACE cxx_std_17)

target_include_directories(slam_vo INTERFACE
  $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)

if(SLAM_VO_ENABLE_WARNINGS)
  target_compile_options(slam_vo INTERFACE
    $<$<CXX_COMPILER_ID:GNU,Clang>:-Wall -Wextra -Wpedantic>
    $<$<CXX_COMPILER_ID:MSVC>:/W4>
  )
endif()

# ---- Dependencies  ----
if(SLAM_VO_REQUIRE_DEPS)
  find_package(OpenCV REQUIRED)
  find_package(Eigen3 REQUIRED)
  find_package(Pangolin REQUIRED)

  # Pangolin target name can vary by install/version
  if(TARGET Pangolin::pangolin)
    set(_PANGOLIN_TARGET Pangolin::pangolin)
  elseif(TARGET pangolin)
    set(_PANGOLIN_TARGET pangolin)
  else()
    message(FATAL_ERROR "Pangolin was found, but no CMake target (Pangolin::pangolin or pangolin) was provided.")
  endif()

  target_link_libraries(slam_vo INTERFACE
    Eigen3::Eigen
    ${OpenCV_LIBS}
    ${_PANGOLIN_TARGET}
  )

  target_include_directories(slam_vo INTERFACE
    ${OpenCV_INCLUDE_DIRS}
  )
endif()

# ---- Add subdirectories only if they exist ----
if(EXISTS "${PROJECT_SOURCE_DIR}/src/CMakeLists.txt")
  add_subdirectory(src)
endif()

if(SLAM_VO_BUILD_TESTS)
  enable_testing()
  if(EXISTS "${PROJECT_SOURCE_DIR}/tests/CMakeLists.txt")
    add_subdirectory(tests)
  endif()
endif()

# ---- Summary ----
message(STATUS "")
message(STATUS "Configuration Summary:")
message(STATUS "  Project:       ${PROJECT_NAME} ${PROJECT_VERSION}")
message(STATUS "  Build Type:    ${CMAKE_BUILD_TYPE}")
message(STATUS "  Build Tests:   ${SLAM_VO_BUILD_TESTS}")
message(STATUS "  Require Deps:  ${SLAM_VO_REQUIRE_DEPS}")
message(STATUS "  Compiler:      ${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}")
message(STATUS "")
