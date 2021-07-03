#include <vector>

template<class MODEL>
void leapfrog(std::size_t dim, std::size_t loop, double epsilon, const MODEL& model,
              std::vector<double> *x, std::vector<double> *p, std::vector<double> *f) {
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
