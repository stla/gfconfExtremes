#' Threshold estimate
#'
#' @param gfi an output of \code{\link{gfigpd}}
#'
#' @return The estimated threshold
#' @export
thresholdEstimate <- function(gfi){
  attr(gfi, "threshold")
}