cmake_minimum_required(VERSION 3.4 FATAL_ERROR)
project(hmc CXX)

set(CMAKE_CXX_STANDARD 17)

find_package(Boost REQUIRED)
include_directories(${Boost_INCLUDE_DIRS})

set(PROGS bimodal harmonic rosenbrock
          bimodal_nuts0 harmonic_nuts0
          bimodal_nuts harmonic_nuts)
foreach(name ${PROGS})
  add_executable(${name} ${name}.cpp)
endforeach(name ${PROGS})
