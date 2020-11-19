#include <Rcpp.h>
#include <boost/math/special_functions/gamma.hpp>
#include <random>
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
const std::vector<size_t> sample_int(const size_t n,
                                     const size_t k,
                                     std::default_random_engine& generator) {
  return takeFirsts(shuffle_n(n, generator), k);
}

const double product(const Rcpp::NumericVector v) {
  double out = 1.0;
  for(auto i = 0; i < v.size(); i++) {
    out *= v(i);
  }
  return out;
}

/* Beta-quantiles for a vector `beta` --------------------------------------- */
Rcpp::NumericVector BetaQuantile(const double g,
                                 const double s,
                                 const double a,
                                 const double prob,
                                 const Rcpp::NumericVector& beta) {
  Rcpp::NumericVector alpha = (1.0 - beta) / prob;
  Rcpp::NumericVector Q;
  if(g == 0.0) {
    Q = a - s * log(alpha);
  } else {
    Q = a + s / g * (pow(alpha, -g) - 1);
  }
  return Q;
}

/* Calculates the Jacobian -------------------------------------------------- */
double Jacobian(const double g,
                const double s,
                const double a,
                const size_t Jnumb,
                const Rcpp::NumericVector& X,
                std::default_random_engine& generator) {
  Rcpp::NumericMatrix Xchoose3(Jnumb, 3);
  const size_t n = X.size();
  for(size_t i = 0; i < Jnumb; i++) {
    const Rcpp::IntegerVector indices = Rcpp::wrap(sample_int(n, 3, generator));
    const Rcpp::NumericVector Xsub = X[indices];
    Xchoose3(i, Rcpp::_) = Xsub;
  }
  const Rcpp::NumericMatrix Xdiff =
      Rcpp::cbind(Xchoose3(Rcpp::_, 1) - Xchoose3(Rcpp::_, 2),
                  Xchoose3(Rcpp::_, 2) - Xchoose3(Rcpp::_, 0),
                  Xchoose3(Rcpp::_, 0) - Xchoose3(Rcpp::_, 1));
  double Jmean;
  if(g == 0.0) {
    Rcpp::NumericVector Jvec(Jnumb);
    for(size_t i = 0; i < Jnumb; i++) {
      Jvec(i) = product(Xdiff(i, Rcpp::_));
    }
    Jmean = Rcpp::mean(Rcpp::abs(Jvec));
  } else {
    Rcpp::NumericMatrix A = 1 + g * (Xchoose3 - a) / s;
    Rcpp::NumericVector Jmat = (log(A) * A / g / g) * Xdiff;
    // Jmat.attr("dim") = Rcpp::Dimension(Jnumb, 3);
    Rcpp::NumericVector Jvec(Jnumb);
    for(size_t i = 0; i < Jnumb; i++) {
      Jvec(i) = Rcpp::sum(Jmat[Rcpp::Range(i * Jnumb, (i + 1) * Jnumb - 1)]);
    }
    Jmean = Rcpp::mean(Rcpp::abs(Jvec));
  }
  return Jmean;
}