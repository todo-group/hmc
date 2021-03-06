#include <random>
#include <iostream>
#include "hmc/nuts0.hpp"
#include "hmc/rosenbrock.hpp"

typedef hmc::rosenbrock model;

int main() {
  int seed = 1234;
  int steps = 1000000;
  int therm = steps / 10;
  double epsilon = 0.1;
  std::mt19937 engine(seed);

  model m;
  hmc::nuts0 mc(m);

  std::vector<double> x(m.dimension(), 0);
  int accepted = 0;
  for (int i = 0; i < (therm + steps); ++i) {
    int len = mc.step(epsilon, m, engine, &x);
    accepted += len;
    if (i >= therm) {
      std::cout << i;
      for (int j = 0; j < m.dimension(); ++j) std::cout << ' ' << x[j];
      std::cout << ' ' << len << std::endl;
    }
  }
  std::cerr << "# average tree size = " << 1.0 * accepted / (therm + steps) << std::endl;
}
