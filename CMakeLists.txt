cmake_minimum_required(VERSION 3.0 FATAL_ERROR)

# Project Name
project(train CXX)

# Find Package
find_package(Torch REQUIRED)
find_package(Boost 1.71.0 REQUIRED COMPONENTS program_options date_time)


# Create Executable File
add_executable(${PROJECT_NAME} src/train.cpp src/arguments/params.cpp src/utils/train_utils/util.cpp src/arguments/params.h src/utils/train_utils/util.h 
    src/scene/gaussian_model.h src/scene/gaussian_model.cpp)

include_directories(
    "${PROJECT_SOURCE_DIR}"
    "${Boost_INCLUDE_DIRS}")

target_link_libraries(${PROJECT_NAME} ${TORCH_LIBRARIES} Boost::program_options Boost::date_time)
set_property(TARGET ${PROJECT_NAME} PROPERTY CXX_STANDARD 17)