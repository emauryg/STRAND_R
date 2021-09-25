#' Main function to run the Variational EM algorithm
#'
#' @param init_pars Initial parameters for the variational EM algorithm
#' @param Y count tensor with dimension 3x3x16x4x96xD, where D is the number of samples
#' @param X design matrix with dimension number of samples x number of covariates
#' @param tau regularization parameter for signature optimization (default: 0.01)
#' @param max_iterEM maximum number of iterations for the overall EM algorithm
#' @param max_iterE max number of iterations for the Expectation step
#' @export
runEM <- function(init_pars, Y, X, tau=0.01, max_iterEM = 30, max_iterE=30){
  ### EM algorithm
  
  gamma_method = "sylvester"
  
  hypLA = list(lr=0.5, max_iter = 1000, tol = list(ratio = 1e-3, abs = 1e-2))
  
  VIparam = list(lambda = init_pars$eta, Delta = init_pars$Delta, Xi = init_pars$Xi, zeta = init_pars$zeta)
  Bparam = list(gamma_sigma = init_pars$gamma_sigma, Sigma = init_pars$Sigma, T0 = init_pars$T0, m = NULL,
                factors = list(bt = init_pars$covs$bt, br = init_pars$covs$br, 
                            epi = init_pars$covs$epi,
                            nuc = init_pars$covs$nuc,
                            clu = init_pars$covs$clu))

  max_elbo = -1e10
  dec_elbo = 0
  patience = 5

  weight_decay = 0.9
  sam_cov = 1
  t1 = Sys.time()
  converged = FALSE
  it = 0
  # ELBO tracker for EM steps
  old_elbo = -1e10
  # ELBO tracker for E or M steps
  old_elbo_ = -1e10
  m_ =  make_m__(Y)
  while(converged == FALSE && it <= max_iterEM){
    it = it +1
    ###########################################
    ## E-step
    ###########################################
    message("E-step:")
    e_converged = FALSE
    it_estep = 0
    if (!is.null(X)){
      # Update variational parameter zeta
      VIparam$zeta <- update_zeta(VIparam$zeta, Bparam$Sigma, Bparam$gamma_sigma, X)

      while(e_converged == FALSE && it_estep <= max_iterE){
        it_estep = it_estep + 1
        # Update variational parameter Xi
        VIparam$Xi <- update_Xi(VIparam$Xi, Bparam$Sigma, 
                                  Bparam$gamma_sigma, X, VIparam$lambda, hypxi, method = gamma_method)
        # Update eta
        laplace_res = update_eta_Delta(Bparam$T0, Bparam$factors, 
                            VIparam$lambda, Bparam$Sigma, Y,VIparam$Xi, X, hypLA)
        VIparam$lambda = laplace_res$eta
        VIparam$Delta = laplace_res$Delta
        
        if(it_estep >= 2){
          elbo_e = compute_elbo(VIparam,Bparam, X, Y)
          e_converged = em_stop(elbo_e, old_elbo_, end = "e.m")
          old_elbo_ = elbo_e
          message(paste("E-step ELBO: ",elbo_e))
        }

      }
    } else{
      ## TODO: incorporate what happens when there are no covariates
    }

    curr = it/(max_iterEM +100)
    hypLA$lr = hypLA$lr * weight_decay^curr 

    ############################
    ## M-step
    ############################
    message("M-step: ")
    if(!is.null(X)){
      Bparam$Sigma = update_Sigma(VIparam$Delta, VIparam$zeta, VIparam$Xi, VIparam$lambda, X, weight = 0.01)
      Bparam$gamma_sigma = update_gamma_sigma(Bparam$gamma_sigma, VIparam$zeta, VIparam$Xi)
    } else {
      ## TODO: need to incorporate what happens when we don't have covariates
    }

    tnf_res = update_TnF(VIparam$lambda, Bparam$factors, Bparam$T0, X, Y, 
                          context= FALSE, missing_rate = m_, weight = 0.01,tau=tau)
    Bparam$T0 = tnf_res$T0
    Bparam$factors = tnf_res$factors

    ## Check for EM convergence after M step
    elbo_em = compute_elbo(VIparam,Bparam, X, Y)

    if (elbo_em > max_elbo){
    max_elbo = elbo_em
    best_VIparam = list(lambda = VIparam$lambda$clone(), Delta = VIparam$Delta$clone(), Xi = VIparam$Xi$clone(), zeta = VIparam$zeta$clone())
    best_Bparam = list(gamma_sigma = Bparam$gamma_sigma$clone(), Sigma = Bparam$Sigma$clone(), T0 = Bparam$T0$clone(), m = make_m__(Y),
              factors = list(bt = Bparam$factors$bt$clone(), br = Bparam$factors$br$clone(), 
                          epi = Bparam$factors$epi$clone(),
                          nuc = Bparam$factors$nuc$clone(),
                          clu = Bparam$factors$clu$clone()))
    }

    if (elbo_em < old_elbo){
    dec_elbo = dec_elbo + 1
      if (dec_elbo == patience){
        message("Cannot be patient any more! \n decided to stop...")
        converged = TRUE
      } else {
            converged = em_stop(elbo_em, old_elbo, end="global")
            old_elbo = elbo_em
      }
    } else {
      converged = em_stop(elbo_em, old_elbo, end="global")
      old_elbo = elbo_em
    }

    message("----------------")
    message(paste("Current EM ELBO:", old_elbo))
    message("-----------------")
  }
  t2 = Sys.time()
  cat("It took: ",difftime(t2,t1, units= "mins")," minutes to converge! \n")
  return(list(VIparam=best_VIparam, Bparam = best_Bparam))
}


em_stop <- function(elbo, old_elbo, end = "e.m"){
  abs_tol = FALSE; rat_tol = FALSE

  if (end == "e.m"){
    if (abs(elbo - old_elbo)/abs(old_elbo) <  1e-3){
      rat_tol = TRUE
    }
    if(abs(elbo - old_elbo) < 2e-1){
      abs_tol = TRUE
    }
  } else if (end=="global"){
    if ( abs(elbo - old_elbo) < 2e-1){
      abs_tol = TRUE
    }
    if( abs(elbo - old_elbo)/abs(old_elbo) < 1e-3){
      rat_tol = TRUE
    }
  }

  if (abs_tol & rat_tol){
    return(TRUE)
  }

  return(FALSE)
}

## Compute elbo for batches of data, due to memory constraints
compute_elbo <- function(VIparam,Bparam, X, Y, batch_size = 32){
  D = nrow(X)
  batch_idx = msplit(1:D, ceiling(D/batch_size))
  elbo = 0
  T0 = Bparam$T0$clone()
  factors = Bparam$factors
  Sigma = Bparam$Sigma$clone()
  Xi = VIparam$Xi$clone()
  gamma_sigma = Bparam$gamma_sigma$clone()
  zeta = VIparam$zeta$clone()
  for(b in batch_idx){
    lambda_b = VIparam$lambda[,b,drop=FALSE]$clone()
    X_b = X[b,,drop=FALSE]$clone()
    Y_b = Y[..,b,drop=FALSE]$clone()
    Delta_b = VIparam$Delta[b,,drop=FALSE]$clone()
    elbo = elbo + compute_elbo_batch(lambda_b, Delta_b,
                                      T0,factors, Sigma,Xi, gamma_sigma,zeta, X_b, Y_b)
  }

  return(elbo/D)
}


compute_elbo_batch <- function(lambda, Delta,T0,factors, Sigma,Xi, gamma_sigma,zeta, X, Y){
  m__ = make_m__(Y)
  p = ncol(X)
  TF = tf(T0, factors, m__)
  yphi_tensor = yphi(lambda, factors, T0, X, Y, context=FALSE, missing_rate=m__)
  SigmaInv = Sigma$inverse()
  elbo = (yphi_tensor$sum(dim=-3)* torch_log(TF+ 1e-14))$sum()

  if(!is.null(X)){
    tr = SigmaInv$matmul(Delta)
    tr = -torch_diagonal(tr, dim1=2, dim2=3)$sum()/2
    elbo = elbo + tr
    mu = Xi$matmul(X$transpose(1,2))

    EqGamma = (mu-lambda)$transpose(1,2)
    EqGamma = EqGamma$matmul(SigmaInv)
    EqGamma = EqGamma$matmul(mu - lambda)
    EqGamma = torch_trace(EqGamma)

    x = X$matmul(zeta)$matmul(X$transpose(1,2))
    x = torch_diagonal(x, dim1=2, dim2=3)$sum(dim=2)
    x = x$dot(torch_diagonal(SigmaInv))

    EqGamma = EqGamma + x 
    
    elbo = elbo - 0.5* EqGamma
    ## torch_slogdet is a more stable way of getting the log of the determinant
    log_det = torch_slogdet(Sigma + 1e-14)[[2]] - torch_slogdet(Delta + 1e-14)[[2]]
    elbo = elbo - 0.5*log_det$sum()

    DivGamma = torch_diagonal(zeta, dim1=2, dim2=3)$sum(dim=2)
    DivGamma = DivGamma/(gamma_sigma^2 + 1e-14)
    DivGamma = DivGamma + (Xi^2)$sum(dim=-1)/ (gamma_sigma^2 + 1e-14)
    DivGamma = DivGamma + 2*p*torch_log(gamma_sigma + 1e-14)
    DivGamma = DivGamma - torch_log(torch_det(zeta)+ 1e-14)

    elbo = elbo - 0.5* DivGamma$sum()
  } else{
    ## TODO: incorporate what happens when there are no covariates
  }

  gc()
  return(elbo$item())

}
