#include <random>
#include <iostream>
#include <gtest/gtest.h>
#include "hmc/rosenbrock.hpp"
#include "hmc/force.hpp"

TEST(RosenbrockTest, RosenbrockTest) {
  int seed = 1234;
  int steps = 10;
  std::mt19937 engine(seed);
  std::uniform_real_distribution<> dist;

  hmc::rosenbrock m;
  int dim = 2;

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

    force_ad_2d(m, x, &f_ad);
    for (int j = 0; j < dim; ++j) {
      EXPECT_DOUBLE_EQ(f_ad[j], f[j]);
    }
  }
}
