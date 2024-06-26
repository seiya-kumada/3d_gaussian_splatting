cmake_minimum_required(VERSION 3.0 FATAL_ERROR)

# Project Name
project(train CXX)

# Find Package
find_package(Torch REQUIRED)
find_package(OpenCV REQUIRED)
find_package(Boost 1.71.0 REQUIRED COMPONENTS program_options date_time)


# Create Executable File
add_executable(${PROJECT_NAME} src/train.cpp src/arguments/params.cpp src/utils/train_utils.cpp 
    src/arguments/params.h src/utils/train_utils.h 
    src/scene/gaussian_model.h src/scene/gaussian_model.cpp
    src/utils/general_utils.h src/utils/general_utils.cpp
    src/scene/scene.h src/scene/scene.cpp
    src/scene/camera.h src/scene/camera.cpp
    src/utils/system_utils.h src/utils/system_utils.cpp
    src/scene/dataset_readers.h src/scene/dataset_readers.cpp
    src/scene/colmap_loader.h src/scene/colmap_loader.cpp
    src/utils/graphics_utils.h src/utils/graphics_utils.cpp
    src/utils/camera_utils.h src/utils/camera_utils.cpp)

target_include_directories(${PROJECT_NAME} PRIVATE src)
option(ENABLE_UNIT_TESTS "Enable unit testing" OFF)
if(ENABLE_UNIT_TESTS)
    add_definitions(-DUNIT_TEST)
    find_package(Boost 1.71.0 REQUIRED COMPONENTS unit_test_framework)
    target_link_libraries(train Boost::unit_test_framework)
endif()

include_directories(
    "${PROJECT_SOURCE_DIR}"
    "${Boost_INCLUDE_DIRS}"
    "${OpenCV_INCLUDE_DIRS}"
    /usr/local/cpplinq)

target_link_libraries(${PROJECT_NAME} ${TORCH_LIBRARIES} Boost::program_options Boost::date_time ${OpenCV_LIBS})
set_property(TARGET ${PROJECT_NAME} PROPERTY CXX_STANDARD 17)
