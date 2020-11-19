#include <random>
#include <Rcpp.h>
#include <boost/math/special_functions/gamma.hpp>
#ifdef _OPENMP
#include <omp.h>
#endif
// [[Rcpp::depends(BH)]]

/* construct vector {0, 1, ..., n-1} ---------------------------------------- */
template <class T>
std::vector<T> integers_n(T n) {
  std::vector<T> out(n);
  for(T i = 0; i < n; i++) {
    out[i] = i;
  }
  return out;
}

/* take k first elements of a vector ---------------------------------------- */
template <class T>
std::vector<T> takeFirsts(const std::vector<T>& v, size_t k) {
  auto first = v.begin();
  auto last = v.begin() + k + 1;
  std::vector<T> out(first, last);
  return out;
}

/* shuffle {0, 1, ..., n-1} ------------------------------------------------- */
const std::vector<size_t> shuffle_n(const size_t n,
                                     std::default_random_engine& generator) {
  std::vector<size_t> elems = integers_n(n);
  std::shuffle(elems.begin(), elems.end(), generator);
  return elems;
}

/* sample k integers among {0, 1, ..., n-1} --------------------------------- */
const std::vector<size_t> sample_int(const size_t n, const size_t k, 
                                    std::default_random_engine& generator) {
  return takeFirsts(shuffle_n(n, generator), k);
}


/* Beta-quantiles for a vector `beta` --------------------------------------- */
Rcpp::NumericVector BetaQuantile(
  double g, double s, double a, double prob, Rcpp::NumericVector beta
){
  Rcpp::NumericVector alpha = (1.0 - beta) / prob;
  Rcpp::NumericVector Q;
  if(g == 0.0){
    Q = a - s * log(alpha);
  }else{
    Q = a + s/g * (pow(alpha, -g) - 1);
  }
  return Q;
}

