// Implementation of No-U-Turn Sampler (Algorithm 3 in Hoffman and Gelman 2011)

#pragma once

#include <iostream>
#include <random>
#include "energy.hpp"
#include "leapfrog.hpp"

namespace hmc {

class nuts0 {
public:
  typedef std::size_t uint_t;
  nuts0(uint_t dim) : dim_(dim), p_(dim_), force_(dim_) {}
  template<class MODEL>
  nuts0(const MODEL& model) : dim_(model.dimension()), p_(dim_), force_(dim_) {}

  template<class MODEL, class RNG>
  uint_t step(uint_t /* loop */, double eps, const MODEL& m, RNG& rng,
              std::vector<double> *x) const {
    return step(eps, m, rng, x);
  }
  template<class MODEL, class RNG>
  uint_t step(double eps, const MODEL& m, RNG& rng, std::vector<double> *x) const {
    std::uniform_real_distribution<double> uniform(0, 1);
    std::normal_distribution<double> gauss(0, 1);
    for (uint_t i = 0; i < dim_; ++i) p_[i] = gauss(rng);
    double logu = std::log(1 - uniform(rng)) - energy(m, *x, p_);
    x_m = *x;
    x_p = *x;

    p_m = p_;
    p_p = p_;
    new_x = *x;
    int n =1, s=1;
    int j =0;
    while (s == 1) {
      double v = uniform(rng);
      int nprime;
      int sprime;
      std::vector<double> xprime;
      if (v < 0.5) {
        std::tie(x_m, p_m, dummy1, dummy2, xprime, nprime, sprime) = buildtree(x_m, p_m, logu, -1, j, eps, m, rng);
      } else {
        std::tie(dummy1, dummy2, x_p, p_p, xprime, nprime, sprime) = buildtree(x_p, p_p, logu, 1, j, eps, m, rng);
      }
      if (sprime == 1) {
        if (uniform(rng) < 1.0 * nprime / n) new_x = xprime;
      }
      n += nprime;
      s = sprime * I(dot(x_p, p_m) - dot(x_m, p_m) >= 0) * I(dot(x_p, p_p) - dot(x_m, p_p) >= 0);
      j++;
    }
    *x = new_x;
    return n;
  }

protected:
  int I(bool st) const {
    if (st) return 1;
    else return 0;
  }

  template<class MODEL, class RNG>
  std::tuple<std::vector<double>, std::vector<double>, std::vector<double>, std::vector<double>, std::vector<double>, int ,int> buildtree(std::vector<double> x, std::vector<double> p, double logu, int v, int j, double eps, const MODEL& m, RNG& rng) const {
    std::uniform_real_distribution<double> uniform(0, 1);
    if (j==0) {
      std::vector<double> xprime(x);
      std::vector<double> pprime(p);
      leapfrog(v * eps, m, &xprime, &pprime, &force_);
      
      std::vector<double> x_m = xprime;
      std::vector<double> p_m = pprime;
      std::vector<double> x_p = xprime;
      std::vector<double> p_p = pprime;
      
      double ene = energy(m, xprime, pprime);
      int nprime = I(logu < -ene);
      int sprime = I(logu < delta_max_ - ene);
      return {x_m, p_m, x_p, p_p, xprime, nprime, sprime};
    } else {
      auto [x_m, p_m, x_p, p_p, xprime, nprime, sprime] = buildtree(x, p, logu, v, j-1, eps, m, rng);
      if (sprime==1) {
        int npp;
        int spp;
        std::vector<double> xpp;
        if (v == -1) {
          std::tie(x_m, p_m, dummy1, dummy2, xpp, npp, spp) = buildtree(x_m, p_m, logu, -1, j - 1, eps, m, rng);
        } else {
          std::tie(dummy1, dummy2, x_p, p_p, xpp, npp, spp) = buildtree(x_p, p_p, logu, 1, j - 1, eps, m, rng);
        }
        if (spp && uniform(rng) < 1.0 * npp / (nprime + npp)) xprime = xpp;
        sprime = spp * I(dot(x_p, p_m) - dot(x_m,p_m) >=0) * I(dot(x_p, p_p) - dot(x_m, p_p) >= 0);
        nprime += npp;     
      }
      return {x_m, p_m, x_p, p_p, xprime, nprime, sprime};
    }
  }

  static double dot(std::vector<double> x, std::vector<double> y) {
    double res = 0;
    for (int i = 0; i < x.size(); i++) res += x[i] * y[i];
    return res;
  }

private:
  static constexpr double delta_max_ = 1000;
  uint_t dim_;
  mutable std::vector<double> p_, force_;
  mutable std::vector<double> x_m, x_p, p_m, p_p, new_x, dummy1, dummy2;
};
  
}
