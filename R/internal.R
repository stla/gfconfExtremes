#' @useDynLib gfiExtremes
#' @importFrom Rcpp evalCpp
NULL

thinChain <- function(chain, skip){
  niterations <- nrow(chain)
  every.ith <- c(TRUE, rep(FALSE, skip))
  keep <- rep(every.ith, ceiling(niterations / (skip+1L)))[1:niterations]
  chain[keep, ]
}

thresholdIndex <- function(Xs, a){
  # Xs is X sorted
  match(TRUE, Xs - a >= 0)
}
