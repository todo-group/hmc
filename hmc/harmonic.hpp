#pragma once

#include <cmath>
#include <vector>
#include <boost/math/differentiation/autodiff.hpp>

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
    auto x0_fvar = boost::math::differentiation::make_fvar<double, 1>(x[0]);
    auto p = potential(x0_fvar);
    (*f)[0] = -p.derivative(1);
  }
};

}
