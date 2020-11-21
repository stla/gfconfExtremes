#' @useDynLib gfconfExtremes
#' @importFrom Rcpp evalCpp
NULL

thinChain <- function(chain, skip){
  number.iterations <- nrow(chain)
  every.ith <- c(TRUE, rep(FALSE, skip))
  eliminate.vector <-
    rep(
      every.ith,
      ceiling((number.iterations) / length(every.ith))
    )[1:number.iterations]
  chain[eliminate.vector, ]
}

#' Title
#'
#' @param X numeric vector of data
#' @param beta vector of probabilities corresponding to the quantiles to be 
#'   estimated
#' @param i index for the initial threshold at the \code{X(i)} order statistic
#' @param gamma.init starting value for gamma in the MCMC
#' @param sigma.init starting value for sigma in the MCMC
#' @param sd.gamma standard deviation for the proposed gamma
#' @param sd.sigma standard deviation for the proposed sigma
#' @param p1 probability that the MCMC will propose a new \code{(gamma,sigma)}; 
#'   \code{(1-p1)} would be the probability that the MCMC chain will propose a 
#'   new index for a new threshold
#' @param p2 probability that the new index proposed will be larger than the 
#'   current index
#' @param lambda1 the small jump the index variable will make
#' @param lambda2 the large jump the index variable will make; happens 1 of 
#'   every 10 iterations
#' @param Jnumb number of subsamples that are taken from the Jacobian
#' @param iter number of iterations per chain (burnin excluded)
#' @param burnin number of the first MCMC iterations discarded
#' @param thin thinning number for the MCMC chain. (e.g. if it is 1 no iteration 
#' is skipped)
#' @param nchains number of MCMC chains to run
#' @param nthreads number of threads to run the chains in parallel
#' @param seeds the seeds used for the MCMC sampler; one seed per chain, or 
#'   \code{NULL} to use random seeds
#'
#' @return
#' @export
#' @importFrom ismev gpd.fit
#' @importFrom doParallel registerDoParallel
#' @importFrom parallel makeCluster stopCluster
#' @importFrom foreach foreach `%dopar%`
#'
#' @examples data("rain", package = "ismev")
#' gf <- gfigpd(rain, beta = c(0.98, 0.99))
gfigpd <- function(
  X, beta, i = floor(0.85 * length(X)), 
  gamma.init = NA, sigma.init = NA, sd.gamma = NA, sd.sigma = NA, 
  p1 = 0.9, p2 = 0.5, lambda1 = 2, lambda2 = 10, Jnumb = 50L, 
  iter = 10000L, burnin = 2000L, thin = 6L,
  nchains = nthreads, nthreads = parallel::detectCores(), seeds = NULL) {
  
  stopifnot(thin >= 1L, nchains >= 1L, nthreads >= 1L)
  nthreads <- min(nthreads, nchains)
  
  X <- sort(X) # -->> so there's no need to sort in C++

  # Intialize the default values for the tuning parameters of the MCMC chain.
  if(is.na(gamma.init) || is.na(sigma.init)) {
    mle.fit <- gpd.fit(X, X[i], show = FALSE)
    if(is.na(gamma.init)) gamma.init <- mle.fit$mle[2L]
    if(is.na(sigma.init)) sigma.init <- mle.fit$mle[1L]
  }
  if(is.na(sd.gamma)) sd.gamma <- 2 * abs(g) / 3
  if(is.na(sd.sigma)) sd.sigma <- 2 * s / 3
  
  skip.number <- thin - 1L
  number.iterations <- (skip.number + 1L) * iter + burnin
  
  if(is.null(seeds)){
    seed1 <- sample.int(2000000L, 1)
    seeds <- seed1 + 2000000L * (0L:(nchains-1L))
  }else{
    if(length(seeds) != nchains){
      stop(
        "Please specify one seed per chain."
      )
    }
    seeds <- abs(as.integer(seeds))
  }
  
  params <- c("gamma", "sigma", "index", paste0("beta", seq_along(beta)))
  
  # run the MCMC chain
  if(nchains == 1L){
    chain <- thinChain(MCMCchain(
      X, beta, gamma.init, sigma.init, i, 
      p1, p2, lambda1, lambda2, sd.gamma, sd.sigma,
      number.iterations, burnin, Jnumb, seeds[1L]
    ), skip.number)
    colnames(chain) <- params
    chain[, 4L:(3L + length(beta))] <- chain[, 4L:(3L + length(beta))] + X[1L]
  }else{
    if(nthreads == 1L){
      chains <- vector("list", nchains)
      for(k in 1L:nchains){
        chains[[k]] <- MCMCchain(
          X, beta, gamma.init, sigma.init, i, 
          p1, p2, lambda1, lambda2, sd.gamma, sd.sigma,
          number.iterations, burnin, Jnumb, seeds[k]
        )
      }
    }else{
      # nblocks <- ceiling(nchains/nthreads)
      # blocks <- vector("list", nblocks)
      # nthreads <- 
      #   c(rep(nthreads, nblocks-1L), nchains - ((nblocks-1L)*nthreads))
      # for(b in 1L:nblocks){
      #   registerDoParallel(cores = nthreads[b])
      # }
      cl <- makeCluster(nthreads)
      registerDoParallel(cl)
      chains <- foreach(
        k = 1L:nchains, .combine = list, .multicombine = TRUE, 
        .export = "MCMCchain"
      ) %dopar% MCMCchain(
        X, beta, gamma.init, sigma.init, i, 
        p1, p2, lambda1, lambda2, sd.gamma, sd.sigma,
        number.iterations, burnin, Jnumb, seeds[k]
      )
      stopCluster(cl)
    }
    chains <- lapply(chains, thinChain, skip = skip.number)
    chains <- lapply(chains, `colnames<-`, value = params)
    chains <- lapply(chains, function(chain){
      chain[, 4L:(3L + length(beta))] <- chain[, 4L:(3L + length(beta))] + X[1L]
      chain
    })
    # threshold per chain ? non, combiner...
  }
  
  if(nchains == 1L){
    out <- coda::as.mcmc(chain)
    attr(out, "threshold") <- NA
  }else{
    out <- as.mcmc.list(chains)
  }
  
  return(out)

  # # Indictor for the acceptance rate. // pas besoin avec coda::rejectionRate
  # ii <- integer(nrow(x.t))
  # for (i in 1L:(nrow(x.t) - 1L)) {
  #   if (x.t[i, 4L] != x.t[i + 1L, 4L]) {
  #     ii[i + 1L] <- 1L
  #   }
  # }
  # acceptance.rate <- mean(ii)
  # cat("acceptance rate: ", acceptance.rate)
  # 
  # x.t[, 4L:(3L + length(beta))] <- x.t[, 4L:(3L + length(beta))] + X[1L]
  # 
  # return(x.t)
}
