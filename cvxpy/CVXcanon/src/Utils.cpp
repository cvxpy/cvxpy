#include "Utils.hpp"
#include <functional>
#include <numeric>

int vecprod(const std::vector<int> &vec) {
  return std::accumulate(vec.begin(), vec.end(), 1.0, std::multiplies<int>());
}

int vecprod_before(const std::vector<int> &vec, int end) {
  return std::accumulate(vec.rbegin() + vec.size() - end, vec.rend(), 1.0, std::multiplies<int>());
}
