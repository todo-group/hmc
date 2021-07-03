#include <random>
#include <iostream>
#include "hmc.hpp"
#include "harmonic.hpp"

typedef harmonic model;

int main() {
  int seed = 1234;
  int steps = 1000000;
  double epsilon = 0.1;
  int loop = 10;
  std::mt19937 engine(seed);

  model m;
  hmc mc(m);

  std::vector<double> x(m.dimension(), 0);
  int accepted = 0;
  for (int i = 0; i < steps; ++i) {
    accepted += mc.step(loop, epsilon, m, engine, &x);
    std::cout << i;
    for (int j = 0; j < m.dimension(); ++j) std::cout << ' ' << x[j];
    std::cout << std::endl;
  }
  std::cerr << "# acceptance rate = " << 1.0 * accepted / steps << std::endl;
}
