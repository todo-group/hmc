#pragma once

#include <algorithm>
#include <vector>

namespace hmc {

template<class MODEL, class T>
T energy(const MODEL& m, const std::vector<T>& x, const std::vector<T>& p) {
  return m.potential(x) + std::inner_product(p.begin(), p.end(), p.begin(), T(0)) / 2;
}

}
