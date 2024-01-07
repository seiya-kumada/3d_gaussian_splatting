cmake_minimum_required(VERSION 3.0 FATAL_ERROR)

# Project Name
project(train CXX)

# Find Package
find_package(Torch REQUIRED)
find_package(Boost 1.71.0 REQUIRED COMPONENTS program_options)


# Create Executable File
add_executable(${PROJECT_NAME} src/main.cpp src/params.cpp src/util.cpp src/params.h src/util.h)

include_directories(
    "${PROJECT_SOURCE_DIR}"
    "${Boost_INCLUDE_DIRS}")

target_link_libraries(${PROJECT_NAME} ${TORCH_LIBRARIES} Boost::program_options)
set_property(TARGET ${PROJECT_NAME} PROPERTY CXX_STANDARD 17)