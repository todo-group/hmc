cmake_minimum_required(VERSION 3.14)
include(cmake/prefix.cmake)
project(hmc CXX)
include(cmake/postfix.cmake)

include_directories(${PROJECT_SOURCE_DIR})

include(googletest)
FetchContent_MakeAvailable(${FetchContents})

find_package(Boost REQUIRED)
include_directories(${Boost_INCLUDE_DIRS})

set(PROGS bimodal harmonic rosenbrock
          bimodal_nuts0 harmonic_nuts0 rosenbrock_nuts0
          bimodal_nuts harmonic_nuts rosenbrock_nuts)
foreach(name ${PROGS})
  add_executable(${name} ${name}.cpp)
endforeach(name ${PROGS})

set(PROGS rosenbrock_force)
foreach(name ${PROGS})
  add_executable(${name} ${name}.cpp)
  target_link_libraries(${name} gtest_main)
  add_test(${name} ${name})
endforeach(name ${PROGS})
