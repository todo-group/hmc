#pragma once

#include <vector>

namespace hmc {

template<class MODEL, class T>
void leapfrog(std::size_t dim, std::size_t loop, T epsilon, const MODEL& model,
              std::vector<T> *x, std::vector<T> *p, std::vector<T> *f /* work */) {
  if (loop > 0) {
    model.force(*x, f);
    for (std::size_t j = 0; j < dim; ++j) (*p)[j] += 0.5 * epsilon * (*f)[j];
    for (std::size_t i = 1; i < loop; ++i) {
      for (std::size_t j = 0; j < dim; ++j) (*x)[j] += epsilon * (*p)[j];
      model.force(*x, f);
      for (std::size_t j = 0; j < dim; ++j) (*p)[j] += epsilon * (*f)[j];
    }
    for (std::size_t j = 0; j < dim; ++j) (*x)[j] += epsilon * (*p)[j];
    model.force(*x, f);
    for (std::size_t j = 0; j < dim; ++j) (*p)[j] += 0.5 * epsilon * (*f)[j];
  }
}

}
