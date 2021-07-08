#pragma once

#include <vector>

namespace hmc {

template<class MODEL, class T>
void leapfrog(std::size_t dim, std::size_t loop, T eps, const MODEL& model,
              std::vector<T> *x, std::vector<T> *p, std::vector<T> *f /* work */) {
  if (loop > 0) {
    model.force(*x, f);
    for (std::size_t j = 0; j < dim; ++j) (*p)[j] += 0.5 * eps * (*f)[j];
    for (std::size_t i = 1; i < loop; ++i) {
      for (std::size_t j = 0; j < dim; ++j) (*x)[j] += eps * (*p)[j];
      model.force(*x, f);
      for (std::size_t j = 0; j < dim; ++j) (*p)[j] += eps * (*f)[j];
    }
    for (std::size_t j = 0; j < dim; ++j) (*x)[j] += eps * (*p)[j];
    model.force(*x, f);
    for (std::size_t j = 0; j < dim; ++j) (*p)[j] += 0.5 * eps * (*f)[j];
  }
}

template<class MODEL, class T>
void leapfrog(std::size_t dim, T eps, const MODEL& model, std::vector<T> *x, std::vector<T> *p, std::vector<T> *f /* work */) {
  leapfrog(dim, 1, eps, model, x, p, f);
}

}
