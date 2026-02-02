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
option(SLAM_VO_REQUIRE_DEPS "Fail configure if OpenCV/Eigen/Pangolin not found" ON)
option(SLAM_VO_ENABLE_WARNINGS "Enable common compiler warnings" ON)

# ---- One project-wide interface target (compile flags + include dirs + deps) ----
add_library(slam_vo INTERFACE)
target_compile_features(slam_vo INTERFACE cxx_std_17)

target_include_directories(slam_vo INTERFACE
        $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
)

if(SLAM_VO_ENABLE_WARNINGS)
  target_compile_options(slam_vo INTERFACE
          $<$<CXX_COMPILER_ID:GNU,Clang,AppleClang>:-Wall -Wextra -Wpedantic>
          $<$<CXX_COMPILER_ID:MSVC>:/W4>
  )
endif()

# ---- Dependencies ----
if(SLAM_VO_REQUIRE_DEPS)
  set(_FIND_MODE REQUIRED)
else()
  set(_FIND_MODE QUIET)
endif()

find_package(Eigen3 ${_FIND_MODE})
find_package(OpenCV ${_FIND_MODE} COMPONENTS core imgproc features2d calib3d highgui)
find_package(Pangolin ${_FIND_MODE} CONFIG)

set(_DEPS_OK TRUE)

if(Eigen3_FOUND)
  target_link_libraries(slam_vo INTERFACE Eigen3::Eigen)
else()
  set(_DEPS_OK FALSE)
endif()

if(OpenCV_FOUND)
  target_link_libraries(slam_vo INTERFACE ${OpenCV_LIBS})
  target_include_directories(slam_vo INTERFACE ${OpenCV_INCLUDE_DIRS})
else()
  set(_DEPS_OK FALSE)
endif()

if(Pangolin_FOUND)
  # Pangolin target name varies by install
  if(TARGET Pangolin::pangolin)
    target_link_libraries(slam_vo INTERFACE Pangolin::pangolin)
  elseif(TARGET Pangolin::Pangolin)
    target_link_libraries(slam_vo INTERFACE Pangolin::Pangolin)
  elseif(TARGET pangolin)
    target_link_libraries(slam_vo INTERFACE pangolin)
  elseif(DEFINED Pangolin_LIBRARIES)
    target_link_libraries(slam_vo INTERFACE ${Pangolin_LIBRARIES})
  else()
    message(WARNING "Pangolin was found but exports no known target; will likely fail to link.")
    set(_DEPS_OK FALSE)
  endif()

  if(DEFINED Pangolin_INCLUDE_DIRS)
    target_include_directories(slam_vo INTERFACE ${Pangolin_INCLUDE_DIRS})
  endif()
else()
  set(_DEPS_OK FALSE)
endif()

# ---- Build the executable only when deps are available (unless you force REQUIRED) ----
if(SLAM_VO_REQUIRE_DEPS OR _DEPS_OK)
  add_executable(slam_vo_run src/main.cpp)
  target_link_libraries(slam_vo_run PRIVATE slam_vo)
else()
  message(STATUS "Dependencies missing -> not building slam_vo_run. Install deps or set SLAM_VO_REQUIRE_DEPS=ON.")
endif()

# ---- Tests ----
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
message(STATUS "  OpenCV:        ${OpenCV_FOUND}")
message(STATUS "  Pangolin:      ${Pangolin_FOUND}")
message(STATUS "  Eigen3:        ${Eigen3_FOUND}")
message(STATUS "  Compiler:      ${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}")
message(STATUS "")
