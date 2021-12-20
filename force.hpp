#pragma once

#include <vector>
#include <boost/math/differentiation/autodiff.hpp>

template<class MODEL, class T>
void force_ad_1d(const MODEL& model, const std::vector<T>& x, std::vector<T> *f) {
  auto x0_fvar = boost::math::differentiation::make_fvar<double, 1>(x[0]);
  auto p = model.potential(x0_fvar);
  (*f)[0] = -p.derivative(1);
}

template<class MODEL, class T>
void force_ad_2d(const MODEL& model, const std::vector<T>& x, std::vector<T> *f) {
  auto const fvar = boost::math::differentiation::make_ftuple<double, 1, 1>(x[0], x[1]);
  auto const& x0_fvar = std::get<0>(fvar);
  auto const& x1_fvar = std::get<1>(fvar);
  auto p = model.potential(x0_fvar, x1_fvar);
  (*f)[0] = -p.derivative(1, 0);
  (*f)[1] = -p.derivative(0, 1);
}
