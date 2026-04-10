cmake_minimum_required(VERSION 3.16)
project(webcam_circle LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenCV 4 REQUIRED
    COMPONENTS core highgui imgcodecs imgproc video videoio)

add_library(webcam_app_lib STATIC webcam_app.cpp tracker.cpp)
target_include_directories(webcam_app_lib PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${OpenCV_INCLUDE_DIRS})
target_link_libraries(webcam_app_lib PUBLIC ${OpenCV_LIBS})
target_compile_options(webcam_app_lib PRIVATE
    -Wall -Wextra -Wpedantic
    $<$<CONFIG:Release>:-O3 -march=native>
    $<$<CONFIG:Debug>:-g -O0>)

add_executable(webcam_circle main.cpp)
target_link_libraries(webcam_circle PRIVATE webcam_app_lib)
target_compile_options(webcam_circle PRIVATE
    -Wall -Wextra -Wpedantic
    $<$<CONFIG:Release>:-O3 -march=native>
    $<$<CONFIG:Debug>:-g -O0 -fsanitize=address>)
target_link_options(webcam_circle PRIVATE
    $<$<CONFIG:Debug>:-fsanitize=address>)

include(FetchContent)
FetchContent_Declare(
    googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG        v1.14.0
)
# Don't let googletest override our build-type flags on Windows (harmless on Linux)
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

# Test executable
enable_testing()

add_executable(tests
    test/webcam_app_test.cpp)
target_include_directories(tests PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/test)
target_link_libraries(tests PRIVATE
    webcam_app_lib
    GTest::gtest_main
    GTest::gmock)
target_compile_options(tests PRIVATE
    -Wall -Wextra
    $<$<CONFIG:Debug>:-g -O0>)

include(GoogleTest)
gtest_discover_tests(tests)
