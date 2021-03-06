#pragma once

#include <vector>

namespace hmc {

struct rosenbrock {
private:
  static constexpr double a = 1;
  static constexpr double b = 5;
public:
  std::size_t dimension() const { return 2; }
  // V = -log W
  template<class T>
  T potential(const std::vector<T>& x) const { return potential(x[0], x[1]); }
  template<class T, class U>
  auto potential(const T& x0, const U& x1) const -> decltype(x0 * x1) {
    return (a - x0) * (a - x0) + b * (x1 - x0 * x0) * (x1 - x0 * x0);
  }
  // F = -grad V
  template<class T>
  void force(const std::vector<T>& x, std::vector<T> *f) const {
    (*f)[0] = 2 * (a - x[0]) + 4 * b * x[0] * (x[1] - x[0] * x[0]);
    (*f)[1] = -2 * b * (x[1] - x[0] * x[0]);
  }
};

}
