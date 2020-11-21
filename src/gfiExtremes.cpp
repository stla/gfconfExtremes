#include <boost/math/special_functions/gamma.hpp>
#include <random>
#include "RcppArmadillo.h"
// [[Rcpp::depends(BH)]]
// [[Rcpp::depends(RcppArmadillo)]]

/*
//~ construct vector {0, 1, ..., n-1} -------------------------------------- ~//
template <class T>
std::vector<T> integers_n(T n) {
  std::vector<T> out(n);
  for(T i = 0; i < n; i++) {
    out[i] = i;
  }
  return out;
}

//~ take k first elements of a vector -------------------------------------- ~//
template <class T>
std::vector<T> takeFirsts(const std::vector<T>& v, size_t k) {
  auto first = v.begin();
  auto last = v.begin() + k + 1;
  std::vector<T> out(first, last);
  return out;
}

//~ shuffle {0, 1, ..., n-1} ----------------------------------------------- ~//
const std::vector<size_t> shuffle_n(const size_t n,
                                    std::default_random_engine& generator) {
  std::vector<size_t> elems = integers_n(n);
  std::shuffle(elems.begin(), elems.end(), generator);
  return elems;
}

//~ sample k integers among {0, 1, ..., n-1} ------------------------------- ~//
const std::vector<size_t> sample_int(const size_t n,
                                     const size_t k,
                                     std::default_random_engine& generator) {
  return takeFirsts(shuffle_n(n, generator), k);
}

//~ sample three integers among {0, 1, ..., n-1} --------------------------- ~//
const std::vector<int> choose3(std::vector<int> elems, const int n,
                               std::default_random_engine& generator) {
  std::uniform_int_distribution<int> sampler1(0, n - 1);
  std::uniform_int_distribution<int> sampler2(0, n - 2);
  std::uniform_int_distribution<int> sampler3(0, n - 3);
  const int i1 = sampler1(generator);
  const int i2 = sampler2(generator);
  const int i3 = sampler3(generator);
  //std::vector<int> elems = integers_n(n); // don't do that in loop!
  elems.erase(elems.begin() + i1);
  const int j2 = elems[i2];
  elems.erase(elems.begin() + i2);
  const int j3 = elems[i3];
  return {i1, j2, j3};
}
*/

//~ sample three integers among {0, 1, ..., n-1} --------------------------- ~//
const std::array<int, 3> choose3(const int n,
                                 std::default_random_engine& generator) {
  std::uniform_int_distribution<int> sampler1(0, n - 1);
  std::uniform_int_distribution<int> sampler2(0, n - 2);
  std::uniform_int_distribution<int> sampler3(0, n - 3);
  const int i1 = sampler1(generator);
  int i2 = sampler2(generator);
  int i3 = sampler3(generator);
  if(i3 == i2)
    i3 = n - 2;
  if(i3 == i1)
    i3 = n - 1;
  if(i2 == i1)
    i2 = n - 1;
  return {i1, i2, i3};
}

//~ sample two integers among {0, 1, ..., n-1} ----------------------------- ~//
const std::array<int, 2> choose2(const int n,
                                 std::default_random_engine& generator) {
  std::uniform_int_distribution<int> sampler1(0, n - 1);
  std::uniform_int_distribution<int> sampler2(0, n - 2);
  const int i1 = sampler1(generator);
  int i2 = sampler2(generator);
  if(i2 == i1) {
    i2 = i1 == n - 2 ? n - 1 : n - 2;
  }
  return {i1, i2};
}

/*
//~ product of vector elements --------------------------------------------- ~//
const double product(const Rcpp::NumericVector v) {
  double out = 1.0;
  for(auto i = 0; i < v.size(); i++) {
    out *= v(i);
  }
  return out;
}

//~ sorts vector ----------------------------------------------------------- ~//
Rcpp::NumericVector stl_sort(const Rcpp::NumericVector x) {
  Rcpp::NumericVector y = clone(x);
  std::sort(y.begin(), y.end());
  return y;
}
*/

//~ Beta-quantiles for a vector `beta` ------------------------------------- ~//
Rcpp::NumericVector BetaQuantile(const double g,
                                 const double s,
                                 const double a,
                                 const double prob,
                                 const Rcpp::NumericVector beta) {
  Rcpp::NumericVector alpha = (1.0 - beta) / prob;
  Rcpp::NumericVector Q;
  if(g == 0.0) {
    Q = a - s * log(alpha);
  } else {
    Q = a + s / g * (pow(alpha, -g) - 1);
  }
  return Q;
}

arma::vec BetaQuantileArma(const double g,
                           const double s,
                           const double a,
                           const double prob,
                           const arma::vec& beta) {
  arma::vec Q;
  if(g == 0.0) {
    Q = a - s * log((1.0 - beta) / prob);
  } else {
    Q = a + s / g * (pow((1.0 - beta) / prob, -g) - 1);
  }
  return Q;
}

//~ Calculates the Jacobian ------------------------------------------------ ~//
double Jacobian(const double g,
                const double s,
                const double a,
                const size_t Jnumb,
                const Rcpp::NumericVector X,
                std::default_random_engine& generator) {
  Rcpp::NumericMatrix Xchoose3(Jnumb, 3);
  const int n = X.size();
  // // Rcpp::Rcout << "n: " << n << "\n";
  for(size_t i = 0; i < Jnumb; i++) {
    const std::array<int, 3> indices = choose3(n, generator);
    // const std::vector<int> indices = {0,1,2};
    // const Rcpp::IntegerVector indices(indices0.begin(), indices0.end());
    //// Rcpp::Rcout << indices[0] << " - " << indices[1] << " - " << indices[2]
    ///<< "\n";
    //  const Rcpp::NumericVector Xsub =
    //  Rcpp::NumericVector::create(X(indices(0)), X(indices(1)),
    //  X(indices(2)));
    // // Rcpp::Rcout << Xsub[0] << " - " << Xsub[1] << " - " << Xsub[2] <<
    // "\n";
    Xchoose3(i, Rcpp::_) = Rcpp::NumericVector::create(
        X(indices[0]), X(indices[1]), X(indices[2]));
  }
  Rcpp::NumericMatrix Xdiff(Jnumb, 3);
  Xdiff(Rcpp::_, 0) = Xchoose3(Rcpp::_, 1) - Xchoose3(Rcpp::_, 2);
  Xdiff(Rcpp::_, 1) = Xchoose3(Rcpp::_, 2) - Xchoose3(Rcpp::_, 0);
  Xdiff(Rcpp::_, 2) = Xchoose3(Rcpp::_, 0) - Xchoose3(Rcpp::_, 1);
  double Jmean;
  if(g == 0.0) {
    const Rcpp::NumericVector Jvec =
        Xdiff(Rcpp::_, 0) * Xdiff(Rcpp::_, 1) * Xdiff(Rcpp::_, 2);
    /*
    Rcpp::NumericVector Jvec(Jnumb);
    for(size_t i = 0; i < Jnumb; i++) {
      Jvec(i) = Xdiff(i, 0) * Xdiff(i, 1) * Xdiff(i, 2);
    }
    */
    Jmean = Rcpp::mean(Rcpp::abs(Jvec));
  } else {
    const Rcpp::NumericMatrix A = g * (Xchoose3 - a) / s;
    Rcpp::NumericVector Jmat0 = (log1p(A) * (1.0 + A) / g / g) * Xdiff;
    const Rcpp::NumericVector Jvec =
        Jmat0[Rcpp::Range(0, Jnumb - 1)] +
        Jmat0[Rcpp::Range(Jnumb, 2 * Jnumb - 1)] +
        Jmat0[Rcpp::Range(2 * Jnumb, 3 * Jnumb - 1)];
    /*
    Rcpp::NumericMatrix Jmat(Jnumb, 3);
    Jmat(Rcpp::_, 0) = Jmat0[Rcpp::Range(0, Jnumb-1)];
    Jmat(Rcpp::_, 1) = Jmat0[Rcpp::Range(Jnumb, 2*Jnumb-1)];
    Jmat(Rcpp::_, 2) = Jmat0[Rcpp::Range(2*Jnumb, 3*Jnumb-1)];
    // Jmat.attr("dim") = Rcpp::Dimension(Jnumb, 3);
    Rcpp::NumericVector Jvec(Jnumb);
    for(size_t i = 0; i < Jnumb; i++) {
      Jvec(i) = Rcpp::sum(Jmat(i, Rcpp::_));
    }
    */
    Jmean = Rcpp::mean(Rcpp::abs(Jvec));
  }
  return Jmean;
}

double JacobianArma(const double g,
                    const double s,
                    const double a,
                    const size_t Jnumb,
                    arma::vec& X,
                    const int n,
                    std::default_random_engine& generator) {
  X = X - a;
  arma::mat Xchoose2(Jnumb, 2);
  if(n >= 250) {
    for(size_t i = 0; i < Jnumb; i++) {
      const std::array<int, 2> indices = choose2(n, generator);
      const arma::rowvec2 row_i = {X.at(indices[0]), X.at(indices[1])};
      Xchoose2.row(i) = row_i;
    }
  }

  double Jmean;
  if(g == 0.0) {
    if(n >= 250) {
      const arma::vec Jvec = Xchoose2.col(0) % Xchoose2.col(1) %
                             (Xchoose2.col(0) - Xchoose2.col(1)) /
                             (2.0 * s * s);
      Jmean = arma::mean(abs(Jvec));
    } else {
      const arma::mat XiXj = (X % X) * X.t();
      // const arma::mat tXiXj = XiXj.t();
      // const arma::mat Ones = arma::ones(n, n);
      const arma::mat UpperTriOnes = arma::trimatu(arma::ones(n, n));
      Jmean = arma::accu(abs(XiXj - XiXj.t()) * UpperTriOnes) /
              (s * s * n * (n - 1));
    }
  } else {
    if(n >= 250) {
      const arma::mat A = g / s * Xchoose2;
      const arma::vec Jvec =
          (Xchoose2.col(0) % (1 + A.col(1)) % log1p(A.col(1)) -
           Xchoose2.col(1) % (1 + A.col(0)) % log1p(A.col(0))) /
          g / g;
      Jmean = arma::mean(abs(Jvec));
    } else {
      const arma::vec A = g / s * X;
      const arma::mat XiXj = X * ((1 + A) % log1p(A)).t();
      const arma::mat UpperTriOnes = arma::trimatu(arma::ones(n, n));
      Jmean = 2 * arma::accu(abs(XiXj - XiXj.t()) * UpperTriOnes) /
              (g * g * n * (n - 1));
    }
  }
  return Jmean;
}

//~ log-density of generalized Pareto XXX ---------------------------------- ~//
const double log_gpd_dens(const double g,
                          const double s,
                          const double a,
                          Rcpp::NumericVector X,
                          const size_t Jnumb,
                          const size_t n,
                          std::default_random_engine& generator) {
  double log_density;
  X = X[X > a];
  // Rcpp::Rcout << "X[X > a]: " << X.size() << "\n";
  const double Max = Rcpp::max(X - a);
  // const double Min = Rcpp::min(X - a); condition '&& Min > 0.0' is always
  // true
  if(s > 0 && g > (-s / Max) && a > 0.0 && g > -0.5) {
    const double J = Jacobian(g, s, a, Jnumb, X, generator);
    // Rcpp::Rcout << "J: " << J << "\n";

    if(g == 0.0) {
      log_density = -1 / s * Rcpp::sum(X - a) + log(J) - n * log(s + a);
    } else {
      log_density = Rcpp::sum((-1 / g - 1) * log1p(g * (X - a) / s)) + log(J) -
                    n * log(s + a);
    }
  } else {
    log_density = -INFINITY;
  }
  // Rcpp::Rcout << "logdens: " << log_density << "\n";

  return log_density;
}

const double log_gpd_densArma(const double g,
                              const double s,
                              const double a,
                              arma::vec& X,
                              const size_t Jnumb,
                              std::default_random_engine& generator) {
  double log_density;
  X = X.elem(find(X > a));
  const int n = X.size();
  const double Max = arma::max(X - a);

  if(s > 0 && g > (-s / Max)) {
    const double J = JacobianArma(g, s, a, Jnumb, X, n, generator);

    if(g == 0.0) {
      log_density = arma::accu(a - X) / s + log(J) - n * log(s);
    } else {
      log_density = arma::accu((-1 / g - 1) * log1p(g * (X - a) / s)) + log(J);
    }
  } else {
    log_density = -INFINITY;
  }

  return log_density;
}

//~ distributions to be sampled -------------------------------------------- ~//
std::uniform_real_distribution<double> uniform(0.0, 1.0);
std::cauchy_distribution<double> cauchy(0.0, 1.0);

//~ propose a new (gamma,sigma) value or a new index for the threshold ----- ~//
std::array<double, 3> MCMCnewpoint(const double g,
                                   const double s,
                                   const double i_dbl,
                                   const double p1,
                                   const double p2,
                                   double lambda,
                                   const double sd_g,
                                   const double sd_s,
                                   const Rcpp::NumericVector X,
                                   const size_t Jnumb,
                                   const int n,
                                   std::default_random_engine& generator,
                                   std::poisson_distribution<int>& poisson1,
                                   std::poisson_distribution<int>& poisson2) {
  // caution with i: zero-index!

  double MHratio, g_star, s_star;

  // Rcpp::Rcout << "i: " << i_dbl << "\n";
  // Rcpp::Rcout << "Xsize: " << X.size() << "\n";

  const int i = (int)(i_dbl + 0.5);
  double a = X(i - 1);
  int i_star;

  if(uniform(generator) > p1) {
    // Rcpp::Rcout << "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA\n";
    int plus_minus = n;
    double dens_pois_star, dens_pois;

    if(uniform(generator) < p2) {
      // std::poisson_distribution<int> poisson(lambda);
      while(plus_minus > n - i - 10) {
        plus_minus = poisson1(generator);
      }
      i_star = i + plus_minus;
      dens_pois_star = p2 / boost::math::gamma_q(n - i - 9, lambda);
      dens_pois = (1.0 - p2) / boost::math::gamma_q(i_star, lambda);
    } else {
      lambda = lambda < i_dbl ? lambda : i_dbl;
      // std::poisson_distribution<int> poisson(lambda);
      while(plus_minus > i - 1) {
        plus_minus = poisson2(generator);
      }
      i_star = i - plus_minus;
      dens_pois_star = (1.0 - p2) / boost::math::gamma_q(i, lambda);
      dens_pois = p2 / boost::math::gamma_q(n - i_star - 9, lambda);
    }

    // Rcpp::Rcout << "istar: " << i_star << "\n";

    const double a_star = X(i_star - 1);
    g_star = g;
    s_star = s + g * (a_star - a);
    MHratio = exp(log_gpd_dens(g_star, s_star, a_star, X, Jnumb, n, generator) -
                  log_gpd_dens(g, s, a, X, Jnumb, n, generator)) *
              dens_pois / dens_pois_star;
  } else {
    // Rcpp::Rcout << "XXXXXX - " << "p1: " << p1 << "\n";
    g_star = g + sd_g * cauchy(generator);
    s_star = s + sd_s * cauchy(generator);
    i_star = i;
    // // Rcpp::Rcout << "gstar" << g_star << "\n";

    MHratio = exp(log_gpd_dens(g_star, s_star, a, X, Jnumb, n, generator) -
                  log_gpd_dens(g, s, a, X, Jnumb, n, generator));
  }

  // // Rcpp::Rcout << "ratio: " << MHratio << "\n";

  std::array<double, 3> newpoint;
  if(!std::isnan(MHratio) && !std::isinf(MHratio) &&
     uniform(generator) < MHratio) {
    newpoint = {g_star, s_star, (double)i_star};
  } else {
    newpoint = {g, s, (double)i};
  }

  return newpoint;
}

arma::vec2 MCMCnewpointArma(const double g,
                            const double s,
                            const double a,
                            const double sd_g,
                            const double sd_s,
                            arma::vec& X,
                            const size_t Jnumb,
                            std::default_random_engine& generator) {
  const double g_star = g + sd_g * cauchy(generator);
  const double s_star = s + sd_s * cauchy(generator);
  const double MHratio =
      exp(log_gpd_densArma(g_star, s_star, a, X, Jnumb, generator) -
          log_gpd_densArma(g, s, a, X, Jnumb, generator));
  arma::vec2 newpoint;
  if(!std::isnan(MHratio) && !std::isinf(MHratio) &&
     uniform(generator) < MHratio) {
    newpoint = {g_star, s_star};
  } else {
    newpoint = {g, s};
  }
}

//~ helper function for MCMCchain ------------------------------------------ ~//
Rcpp::NumericVector concat(const double g,
                           const double s,
                           const double i,
                           const Rcpp::NumericVector beta,
                           const size_t lbeta) {
  Rcpp::NumericVector out(3 + lbeta);
  out(0) = g;
  out(1) = s;
  out(2) = i;
  for(size_t k = 3; k < 3 + lbeta; k++) {
    out(k) = beta(k - 3);
  }
  return out;
}

//~ function that runs the MCMC chain -------------------------------------- ~//
// [[Rcpp::export]]
Rcpp::NumericMatrix MCMCchain(Rcpp::NumericVector X,
                              const Rcpp::NumericVector beta,
                              const double g,
                              const double s,
                              const double a,
                              const int i,
                              const double p1,
                              const double p2,
                              const double lambda1,
                              const double lambda2,
                              const double sd_g,
                              const double sd_s,
                              const size_t niter,
                              const size_t nburnin,  // almost not used here
                              const size_t Jnumb,
                              const unsigned seed) {
  std::default_random_engine generator(seed);

  // X = stl_sort(X); sorted in R
  X = X - X(0);
  const size_t lbeta = beta.size();
  const double i_dbl = (double)i;
  const int n = X.size();

  Rcpp::NumericMatrix xt(niter, 3 + lbeta);
  xt(0, Rcpp::_) = concat(g, s, i_dbl,  // caution with X(i) !!
                          BetaQuantile(g, s, a, 1.0 - i_dbl / n, beta), lbeta);

  std::poisson_distribution<int> poisson1(lambda1);
  std::poisson_distribution<int> poisson2(lambda2);
  std::poisson_distribution<int> poisson3(i_dbl);

  double lambda;

  for(size_t j = 0; j < niter - 1; j++) {
    bool b = j % 10 == 0;
    lambda = b ? lambda2 : lambda1;
    std::array<double, 3> gsi;
    if(lambda < i_dbl) {
      if(b) {
        gsi = MCMCnewpoint(xt(j, 0), xt(j, 1), xt(j, 2), p1, p2, lambda, sd_g,
                           sd_s, X, Jnumb, n, generator, poisson2, poisson2);
      } else {
        gsi = MCMCnewpoint(xt(j, 0), xt(j, 1), xt(j, 2), p1, p2, lambda, sd_g,
                           sd_s, X, Jnumb, n, generator, poisson1, poisson1);
      }
    } else {
      if(b) {
        gsi = MCMCnewpoint(xt(j, 0), xt(j, 1), xt(j, 2), p1, p2, lambda, sd_g,
                           sd_s, X, Jnumb, n, generator, poisson2, poisson3);
      } else {
        gsi = MCMCnewpoint(xt(j, 0), xt(j, 1), xt(j, 2), p1, p2, lambda, sd_g,
                           sd_s, X, Jnumb, n, generator, poisson1, poisson3);
      }
    }
    xt(j + 1, Rcpp::_) =
        concat(gsi[0], gsi[1], gsi[2],  // caution with X(i) !!
               BetaQuantile(gsi[0], gsi[1], X((int)(gsi[2] + 0.5) - 1),
                            1.0 - gsi[2] / n, beta),
               lbeta);
  }

  xt = xt(Rcpp::Range(nburnin, niter - 1), Rcpp::_);

  return xt;
  /*
  niter = xt.nrow();
  Rcpp::IntegerVector every_ith = Rcpp::rep(0, nskip+1);
  every_ith(0) = 1;
  Rcpp::IntegerVector eliminate =
    Rcpp::rep(every_ith, (int)ceil((double)niter / (nskip+1)));
  eliminate = eliminate[Rcpp::Range(0,niter)];
  xt = xt[eliminate, Rcpp::_];
  */
}