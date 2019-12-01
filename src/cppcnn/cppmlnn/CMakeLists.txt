cmake_minimum_required(VERSION 3.0)

# Project name
project(cppmlnn)

include(CTest)

set (CMAKE_CXX_STANDARD 11)

# The version number
set(cppmlnn_VERSION_MAJOR 1)
set(cppmlnn_VERSION_MINOR 0)

if(MSVC)
  add_compile_options(/Zi)
else()
  add_compile_options(-g -m64)
endif()

file(GLOB HEADERS *.h)
file(GLOB HEADERS_UTILS ../utils/*.h)

aux_source_directory("." SRCS)
aux_source_directory("../../utils" SRCS_UTILS)

set(TARGET_NAME cppmlnn)
add_executable(${TARGET_NAME})
target_sources(${TARGET_NAME} PRIVATE ${HEADERS} ${HEADERS_UTILS} ${SRCS} ${SRCS_UTILS})
target_include_directories(${TARGET_NAME} PRIVATE "../../../3rdparty/")
target_include_directories(${TARGET_NAME} PRIVATE "../../utils")

set_target_properties( ${TARGET_NAME}
    PROPERTIES
    ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)

include("../../../cmake_modules/find-openmp.cmake")