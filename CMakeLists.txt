cmake_minimum_required(VERSION 3.14)
include(cmake/prefix.cmake)
project(hmc CXX)
include(cmake/postfix.cmake)

include_directories(${PROJECT_SOURCE_DIR})

include(googletest)
FetchContent_MakeAvailable(${FetchContents})

find_package(Boost)
if(Boost_FOUND)
  include_directories(${Boost_INCLUDE_DIRS})
endif(Boost_FOUND)

add_subdirectory(test)
