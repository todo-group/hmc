if(NOT CMAKE_C_COMPILER)
  set(CMAKE_C_COMPILER "icc" CACHE STRING "" FORCE)
endif(NOT CMAKE_C_COMPILER)
if(NOT CMAKE_CXX_COMPILER)
  set(CMAKE_CXX_COMPILER "icpc" CACHE STRING "" FORCE)
endif(NOT CMAKE_CXX_COMPILER)
if(NOT CMAKE_Fortran_COMPILER)
  set(CMAKE_Fortran_COMPILER "ifort" CACHE STRING "" FORCE)
endif(NOT CMAKE_Fortran_COMPILER)