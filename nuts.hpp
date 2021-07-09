// Implementation of No-U-Turn Sampler (equivalent to Algorithm 3 in Hoffman and Gelman 2011)
//
// * Avoid recursion
// * Avoid allocation/free at every step

#pragma once

#include <iostream>
#include <random>
#include "energy.hpp"
#include "leapfrog.hpp"

namespace hmc {

class nuts {
public:
  typedef std::size_t uint_t;
  nuts(uint_t dim) : dim_(dim), p_(dim_), force_(dim_), tree_(0, basetree(dim_)) {}
  template<class MODEL>
  nuts(const MODEL& model) : dim_(model.dimension()), p_(dim_), force_(dim_),
                             tree_(0, basetree(dim_)) {}

  template<class MODEL, class RNG>
  uint_t step(uint_t /* loop */, double eps, const MODEL& m, RNG& rng,
              std::vector<double> *x) const {
    std::uniform_real_distribution<double> uniform(0, 1);
    std::normal_distribution<double> gauss(0, 1);
    for (uint_t i = 0; i < dim_; ++i) p_[i] = gauss(rng);
    double logu = std::log(1 - uniform(rng)) - energy(m, *x, p_);
    check_capacity(1);
    tb_ = 0;
    tree_[0].init(*x, p_);

    uint_t j = 0;
    bool to_continue = true;
    while (to_continue) {
      if (tb_ != 0) std::cerr << "error\n";
      check_capacity(j + 2);
      int v = (uniform(rng) > 0.5) ? 1 : -1; // v = 1 (-1) for going right (left)
      tree_[tb_].copy(v, x, &p_);
      for (uint_t i = 0; i < (1 << j); ++i) {
        leapfrog(v * eps, m, x, &p_, &force_);
        double ene = energy(m, *x, p_);
        if (logu > delta_max_ - ene) { to_continue = false; break; }
        tree_[++tb_].init(*x, p_, (logu < -ene));
        if (!(to_continue = merge(v, rng))) break;
      }
      ++j;
    }
    std::copy(tree_[0].x_next.begin(), tree_[0].x_next.end(), x->begin());
    return tree_[0].n;
  }

protected:
  struct basetree {
    basetree(std::size_t dim) : x_plus(dim), p_plus(dim), x_minus(dim), p_minus(dim), x_next(dim) {}
    void init(const std::vector<double>& x, const std::vector<double>& p, bool valid = true) {
      std::copy(x.begin(), x.end(), x_plus.begin());
      std::copy(p.begin(), p.end(), p_plus.begin());
      std::copy(x.begin(), x.end(), x_minus.begin());
      std::copy(p.begin(), p.end(), p_minus.begin());
      std::copy(x.begin(), x.end(), x_next.begin());
      level = 0;
      n = valid;
    }
    void copy(int v, std::vector<double> *x, std::vector<double> *p) const {
      if (v == 1) {
        std::copy(x_plus.begin(), x_plus.end(), x->begin());
        std::copy(p_plus.begin(), p_plus.end(), p->begin());
      } else{
        std::copy(x_minus.begin(), x_minus.end(), x->begin());
        std::copy(p_minus.begin(), p_minus.end(), p->begin());
      }
    }
    bool not_turn() const {
      std::size_t n = x_plus.size();
      double prod_p = 0;
      double prod_m = 0;
      for (std::size_t i = 0; i < n; ++i) {
        double xpm = x_plus[i] - x_minus[i];
        prod_p += xpm * p_plus[i];
        prod_m += xpm * p_minus[i];
      }
      return (prod_p > 0) && (prod_m > 0);
    }
    std::size_t level, n;
    std::vector<double> x_plus, p_plus, x_minus, p_minus, x_next;
  };
  
  template<class RNG>
  bool merge(int v, RNG& rng) const {
    std::uniform_real_distribution<double> uniform(0, 1);
    bool not_turn = true;
    while (tb_ > 0 && tree_[tb_].level == tree_[tb_-1].level) {
      if (not_turn) {
        if (v == 1) {
          std::swap(tree_[tb_-1].x_plus, tree_[tb_].x_plus);
          std::swap(tree_[tb_-1].p_plus, tree_[tb_].p_plus);
        } else {
          std::swap(tree_[tb_-1].x_minus, tree_[tb_].x_minus);
          std::swap(tree_[tb_-1].p_minus, tree_[tb_].p_minus);
        }
        if (tb_ == 1) {
          if (uniform(rng) < 1.0 * tree_[tb_].n / tree_[tb_-1].n) {
            std::swap(tree_[tb_-1].x_next, tree_[tb_].x_next);
          }
        } else {
          if (uniform(rng) < 1.0 * tree_[tb_].n / (tree_[tb_-1].n + tree_[tb_].n)) {
            std::swap(tree_[tb_-1].x_next, tree_[tb_].x_next);
          }
        }
        tree_[tb_-1].n += tree_[tb_].n;
        ++tree_[tb_-1].level;
        --tb_;
        not_turn &= tree_[tb_].not_turn();
      } else {
        tree_[tb_-1].n += tree_[tb_].n;
        ++tree_[tb_-1].level;
        --tb_;
      }
    }
    return not_turn;
  }
  
  void check_capacity(uint_t n) const {
    while (tree_.size() < n) tree_.push_back(basetree(dim_));
  }
  
private:
  static constexpr double delta_max_ = 1000;
  uint_t dim_;
  mutable uint_t tb_; // index of the last tree block
  mutable std::vector<double> p_, force_;
  mutable std::vector<basetree> tree_;
};
  
}
