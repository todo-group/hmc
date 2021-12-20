#pragma once

#include <cmath>
#include <vector>

namespace hmc {

struct harmonic {
public:
  std::size_t dimension() const { return 1; }
  // V = -log(W)
  template<class T>
  T potential(const std::vector<T>& x) const { return potential(x[0]); }
  template<class T>
  T potential(const T& x0) const { return x0 * x0; }
  // F = -grad V
  template<class T>
  void force(const std::vector<T>& x, std::vector<T> *f) const {
    (*f)[0] = -2 * x[0];
  }
};

}
