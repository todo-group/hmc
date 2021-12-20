#include <random>
#include <iostream>
#include <gtest/gtest.h>
#include "hmc/harmonic.hpp"
#include "hmc/force.hpp"

TEST(harmonic_force, test0) {
  int seed = 1234;
  int steps = 10;
  std::mt19937 engine(seed);
  std::uniform_real_distribution<> dist(-0.5, 0.5);

  hmc::harmonic m;
  int dim = 1;

  std::vector<double> x(dim);
  std::vector<double> f(dim), f_ad(dim);
  for (int i = 0; i < steps; ++i) {
    for (int j = 0; j < dim; ++j) {
      x[j] = dist(engine);
      std::cout << x[j] << ' ';
    }
    m.force(x, &f);
    for (int j = 0; j < dim; ++j) {
      std::cout << f[j] << ' ';
    }
    std::cout << std::endl;

    force_ad_1d(m, x, &f_ad);
    for (int j = 0; j < dim; ++j) {
      EXPECT_DOUBLE_EQ(f_ad[j], f[j]);
    }
  }
}
