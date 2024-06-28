# 1. cmake version
cmake_minimum_required(VERSION 3.2)

# 2. project name
PROJECT(test_share)

# 3. head file path
INCLUDE_DIRECTORIES(
    include
)

# 4. source directory
AUX_SOURCE_DIRECTORY(src DIR_SRCS)

AUX_SOURCE_DIRECTORY(include INCLUDE_DIR)

# 5. set environment variable
SET(TEST_MATH
${DIR_SRCS}
)

SET(INC_DIR
${INCLUDE_DIR}
)

message(STATUS "the PROJECT_NAME is ${PROJECT_NAME}")

add_library(example SHARED src/libexample.cpp)

# 6. add exectable file
ADD_EXECUTABLE(${PROJECT_NAME} ${TEST_MATH})

# 7. add link library
TARGET_LINK_LIBRARIES(${PROJECT_NAME})

target_include_directories(${PROJECT_NAME} PUBLIC ${INC_DIR})
