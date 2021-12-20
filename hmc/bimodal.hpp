#pragma once

#include <cmath>
#include <vector>

namespace hmc {

struct bimodal {
  static constexpr double cx0 = 3;
  static constexpr double cy0 = 2;
  static constexpr double s0 = 1;
  static constexpr double w0 = 0.7;
  static constexpr double cx1 = -1;
  static constexpr double cy1 = -0.5;
  static constexpr double s1 = 0.5;
  static constexpr double w1 = 0.3;
public:
  std::size_t dimension() const { return 2; }
  // V = -log W
  template<class T>
  T potential(const std::vector<T>& x) const { return potential(x[0], x[1]); }
  template<class T, class U>
  auto potential(const T& x0, const U& x1) const -> decltype(x0 * x1) {
    using std::log;
    return -log(gauss_2d(x0, x1, w0, cx0, cy0, s0) + gauss_2d(x0, x1, w1, cx1, cy1, s1));
  }
  // F = -grad V
  template<class T>
  void force(const std::vector<T>& x, std::vector<T> *f) const {
    auto x0 = x[0];
    auto x1 = x[1];
    auto z = gauss_2d(x0, x1, w0, cx0, cy0, s0) + gauss_2d(x0, x1, w1, cx1, cy1, s1);
    (*f)[0] = 2 * (gauss_2d_dx(x0, x1, w0, cx0, cy0, s0) + gauss_2d_dx(x0, x1, w1, cx1, cy1, s1)) / z;
    (*f)[1] = 2 * (gauss_2d_dy(x0, x1, w0, cx0, cy0, s0) + gauss_2d_dy(x0, x1, w1, cx1, cy1, s1)) / z;
  }
protected:
  template<class T, class U>
  auto gauss_2d(T x, U y, double w, double cx, double cy, double sig) const -> decltype(x * y) {
    using std::exp;
    return w * exp(-((x - cx) * (x - cx) + (y - cy) * (y - cy)) / (2 * sig * sig))
      / (sig * sqrt(2 * M_PI));
  }
  template<class T, class U>
  auto gauss_2d_dx(T x, U y, double w, double cx, double cy, double sig) const -> decltype(x * y) {
    using std::exp;
    return -w * ((x-cx) / (2 * sig * sig)) * exp(-((x - cx) * (x - cx) + (y - cy) * (y - cy)) / (2 * sig * sig))
      / (sig * sqrt(2 * M_PI));
  }
  template<class T, class U>
  auto gauss_2d_dy(T x, U y, double w, double cx, double cy, double sig) const -> decltype(x * y) {
    using std::exp;
    return -w * ((y-cy) / (2 * sig * sig)) * exp(-((x - cx) * (x - cx) + (y - cy) * (y - cy)) / (2 * sig * sig))
      / (sig * sqrt(2 * M_PI));
  }
};

}
