#include <iostream>
#include <random>
#include "leapfrog.hpp"

class hmc {
public:
  hmc(std::size_t dim) : dim_(dim), x_(dim_), p_(dim_), f_(dim_) {}
  template<class MODEL>
  hmc(const MODEL& model) : dim_(model.dimension()), x_(dim_), p_(dim_), f_(dim_) {}
  template<class MODEL, class RNG>
  bool step(std::size_t loop, double epsilon, const MODEL& model, RNG& rng,
            std::vector<double> *x_in) {
    std::uniform_real_distribution<double> uniform(0, 1);
    std::normal_distribution<double> gauss(0, 1);
    std::copy(x_in->begin(), x_in->end(), x_.begin());
    double ene = model.potential(x_);
    for (std::size_t i = 0; i < dim_; ++i) {
      p_[i] = gauss(rng);
      ene += p_[i] * p_[i] / 2;
    }
    leapfrog(dim_, loop, epsilon, model, &x_, &p_, &f_);
    double ene_new = model.potential(x_);
    for (std::size_t i = 0; i < dim_; ++i) {
      ene += p_[i] * p_[i] / 2;
    }
    if (uniform(rng) < std::exp(-(ene_new - ene))) {
      std::swap(*x_in, x_);
      return true;
    } else {
      return false;
    }
  }
private:
  std::size_t dim_;
  std::vector<double> x_, p_, f_;
};
