set(PROGS bimodal_hmc harmonic_hmc rosenbrock_hmc
          bimodal_nuts0 harmonic_nuts0 rosenbrock_nuts0
          bimodal_nuts harmonic_nuts rosenbrock_nuts)
foreach(name ${PROGS})
  add_executable(${name} ${name}.cpp)
endforeach(name ${PROGS})

if(Boost_FOUND)
  set(PROGS bimodal_force harmonic_force rosenbrock_force)
  foreach(name ${PROGS})
    add_executable(${name} ${name}.cpp)
    target_link_libraries(${name} gtest_main)
    add_test(${name} ${name})
  endforeach(name ${PROGS})
endif(Boost_FOUND)
