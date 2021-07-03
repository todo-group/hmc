#include <vector>
#include <boost/math/differentiation/autodiff.hpp>

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
    auto const fvar = boost::math::differentiation::make_ftuple<double, 1, 1>(x[0], x[1]);
    auto const& x0_fvar = std::get<0>(fvar);
    auto const& x1_fvar = std::get<1>(fvar);
    auto p = potential(x0_fvar, x1_fvar);
    (*f)[0] = -p.derivative(1, 0);
    (*f)[1] = -p.derivative(0, 1);
  }
};
