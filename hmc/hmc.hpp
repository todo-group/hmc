#pragma once

#include <iostream>
#include <random>
#include "energy.hpp"
#include "leapfrog.hpp"

namespace hmc {

class hmc {
public:
  hmc(std::size_t dim) { init(dim); }
  template<class MODEL>
  hmc(const MODEL& model) { init(model.dimension()); }

  void init(std::size_t dim) {
    dim_ = dim;
    x_.resize(dim_);
    p_.resize(dim_);
    f_.resize(dim_);
  }

  template<class MODEL, class RNG>
  bool step(std::size_t loop, double eps, const MODEL& model, RNG& rng,
            std::vector<double> *x_in) const {
    std::uniform_real_distribution<double> uniform(0, 1);
    std::normal_distribution<double> gauss(0, 1);

    // initial condition
    std::copy(x_in->begin(), x_in->end(), x_.begin());
    for (std::size_t i = 0; i < dim_; ++i) p_[i] = gauss(rng);
    double ene = energy(model, x_, p_);

    leapfrog(loop, eps, model, &x_, &p_, &f_);

    // metropolis filter
    double ene_new = energy(model, x_, p_);
    if (uniform(rng) < std::exp(-(ene_new - ene))) {
      std::swap(*x_in, x_);
      return true;
    } else {
      return false;
    }
  }

private:
  std::size_t dim_;
  mutable std::vector<double> x_, p_, f_;
};

}
