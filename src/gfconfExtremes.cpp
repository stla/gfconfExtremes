#include <random>
#include <Rcpp.h>
#include <boost/math/special_functions/gamma.hpp>
#ifdef _OPENMP
#include <omp.h>
#endif
// [[Rcpp::depends(BH)]]

/* construct vector {0, 1, ..., n-1} ---------------------------------------- */
template <class T>
std::vector<T> zero2n(T n) {
  std::vector<T> out(n);
  for(T i = 0; i < n; i++) {
    out[i] = i;
  }
  return out;
}

/* sample with repla {0, 1, ..., n-1} ---------------------------------------- */
const std::vector<size_t> sample_int(const size_t n,
                                     std::default_random_engine& generator) {
  std::vector<size_t> elems = zero2n(n);
  std::shuffle(elems.begin(), elems.end(), generator);
  return elems;
}


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

