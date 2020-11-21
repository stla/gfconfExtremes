// Generated by using Rcpp::compileAttributes() -> do not edit by hand
// Generator token: 10BE3573-1514-4C36-9D1C-5A225CD40393

#include <Rcpp.h>

using namespace Rcpp;

// MCMCchain
Rcpp::NumericMatrix MCMCchain(Rcpp::NumericVector X, const Rcpp::NumericVector beta, const double g, const double s, const double a, const int i, const double p1, const double p2, const double lambda1, const double lambda2, const double sd_g, const double sd_s, const size_t niter, const size_t nburnin, const size_t Jnumb, const unsigned seed);
RcppExport SEXP _gfiExtremes_MCMCchain(SEXP XSEXP, SEXP betaSEXP, SEXP gSEXP, SEXP sSEXP, SEXP aSEXP, SEXP iSEXP, SEXP p1SEXP, SEXP p2SEXP, SEXP lambda1SEXP, SEXP lambda2SEXP, SEXP sd_gSEXP, SEXP sd_sSEXP, SEXP niterSEXP, SEXP nburninSEXP, SEXP JnumbSEXP, SEXP seedSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< Rcpp::NumericVector >::type X(XSEXP);
    Rcpp::traits::input_parameter< const Rcpp::NumericVector >::type beta(betaSEXP);
    Rcpp::traits::input_parameter< const double >::type g(gSEXP);
    Rcpp::traits::input_parameter< const double >::type s(sSEXP);
    Rcpp::traits::input_parameter< const double >::type a(aSEXP);
    Rcpp::traits::input_parameter< const int >::type i(iSEXP);
    Rcpp::traits::input_parameter< const double >::type p1(p1SEXP);
    Rcpp::traits::input_parameter< const double >::type p2(p2SEXP);
    Rcpp::traits::input_parameter< const double >::type lambda1(lambda1SEXP);
    Rcpp::traits::input_parameter< const double >::type lambda2(lambda2SEXP);
    Rcpp::traits::input_parameter< const double >::type sd_g(sd_gSEXP);
    Rcpp::traits::input_parameter< const double >::type sd_s(sd_sSEXP);
    Rcpp::traits::input_parameter< const size_t >::type niter(niterSEXP);
    Rcpp::traits::input_parameter< const size_t >::type nburnin(nburninSEXP);
    Rcpp::traits::input_parameter< const size_t >::type Jnumb(JnumbSEXP);
    Rcpp::traits::input_parameter< const unsigned >::type seed(seedSEXP);
    rcpp_result_gen = Rcpp::wrap(MCMCchain(X, beta, g, s, a, i, p1, p2, lambda1, lambda2, sd_g, sd_s, niter, nburnin, Jnumb, seed));
    return rcpp_result_gen;
END_RCPP
}

static const R_CallMethodDef CallEntries[] = {
    {"_gfiExtremes_MCMCchain", (DL_FUNC) &_gfiExtremes_MCMCchain, 16},
    {NULL, NULL, 0}
};

RcppExport void R_init_gfiExtremes(DllInfo *dll) {
    R_registerRoutines(dll, NULL, CallEntries, NULL, NULL);
    R_useDynamicSymbols(dll, FALSE);
}
