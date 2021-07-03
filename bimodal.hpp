#include <cmath>
#include <vector>
#include <boost/math/differentiation/autodiff.hpp>

struct bimodal {
private:
  static constexpr double cx0 = 3;
  static constexpr double cy0 = 2;
  static constexpr double s0 = 1;
  static constexpr double w0 = 0.7;
  static constexpr double cx1 = -1;
  static constexpr double cy1 = -0.5;
  static constexpr double s1 = 0.5;
  static constexpr double w1 = 1 - w0;
public:
  std::size_t dimension() const { return 2; }
  // V = -log W
  template<class T>
  T potential(const std::vector<T>& x) const { return potential(x[0], x[1]); }
  template<class T, class U>
  auto potential(const T& x0, const U& x1) const -> decltype(x0 * x1) {
    using std::log;
    return -log(w0 * gauss_2d(x0, x1, cx0, cy0, s0) + w1 * gauss_2d(x0, x1, cx1, cy1, s1));
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
protected:
  template<class T, class U>
  auto gauss_2d(T x, U y, double cx, double cy, double sig) const -> decltype(x * y) {
    using std::exp;
    return exp(-((x - cx) * (x - cx) + (y - cy) * (y - cy)) / (2 * sig * sig))
      / (sig * sqrt(2 * M_PI));
  }
};
