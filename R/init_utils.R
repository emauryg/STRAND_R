## Initialization functions

### Random Initialize

# if (cuda_is_available()) {
#    device <- torch_device("cuda")
# } else {
#    device <- torch_device("cpu")
# }

# library(MCMCpack)
# library(mvtnorm)

Categorical <- function(n_samples, probs){
  ## assumes inputs in torch tensor format
  n_samples = n_samples$item()
  probs = as_array(probs$cpu())
  K = length(probs)
  tmp = rmultinom(n_samples, 1, prob = probs)
  tmp = colSums(c(1:K)*tmp)
  tmp = as.integer(tmp) #torch_tensor(tmp, dtype = torch_long(), device=device)
  return(tmp)
}


#' Generate sample data from the structural topic model framework
#'
#' @param V number of trinucleotide contents (i.e. 96)
#' @param K number of signatures or latent factors (i.e. 5)
#' @param D number samples (i.e. 100)
#' @param p number of covariates (i.e. 3)
#' @param gamma_mean mean of effect sizes of covariates in the odds ratio scale. 
#' @return list of parameters and sample data in torch tensor format.
#' @export
generate_data <- function(V,K,D,p,no_covars=FALSE, gamma_mean = 0){

  ## Generate multiplicative effects
  gen_b <- function(K){
    bt = t(rdirichlet(K, 5*c(0.6, 0.4)))
    bt = torch_tensor(bt, device=device)

    br = t(rdirichlet(K, 5*c(0.4,0.6)))
    br = torch_tensor(br, device=device)
    return(list(bt=bt, br = br))
  }

  ## generate k_epi
  gen_epi <- function(K, a = 30){
    epi = t(rdirichlet(K,rep(0.6*a,16)))
    epi = torch_tensor(epi,device=device)
    return(epi)
  }

  ## Generate k_nuc
  gen_nuc <- function(K, a = 10){
    nuc = t(rdirichlet(K,c(0.2,0.6,0.1,0.1)*a))
    nuc = torch_tensor(nuc, device=device)
    return(nuc)
  }

  ## Generat k_clu
  gen_clu <- function(K, a=1, H = c(0.8, 0.2)){
    clu = t(rdirichlet(K,H*a))
    clu = torch_tensor(clu, device=device)
  }

  ## Generating transcription and translation effects
  gen_T <- function(V, K, a = 100){

    tmp = t(rdirichlet(K, rep(0.3, V)))
    atmp = a*tmp
    
    T0_CL = torch_tensor(apply(atmp, 2, rdirichlet, n=1), device=device)
    T0_CG = torch_tensor(apply(atmp, 2, rdirichlet, n=1), device=device)
    T0_TL = torch_tensor(apply(atmp, 2, rdirichlet, n=1), device=device)
    T0_TG = torch_tensor(apply(atmp, 2, rdirichlet, n=1), device=device)

    # T0_CL = torch_tensor(t(rdirichlet(K, rep(0.8,V))), device=device)
    # T0_CG = torch_tensor(t(rdirichlet(K, rep(0.8,V))), device=device)
    # T0_TL = torch_tensor(t(rdirichlet(K, rep(0.8,V))), device=device)
    # T0_TG = torch_tensor(t(rdirichlet(K, rep(0.8,V))), device=device)

    T0 = torch_stack(c(T0_CL, T0_CG, T0_TL, T0_TG))$reshape(c(2,2,V, K))
    return(T0)
  }

  ## Generate missing
  gen_m <- function(missing_prob, a = 10){
    m_ = torch_tensor(t(rdirichlet(1, missing_prob*a)), device=device)
  }

  sample_count_from_nb <- function(n, mean, n_sample){
    prob = (n/(n+mean))
    tmp = torch_tensor(rnbinom(size=n, n = n_sample, prob=prob), device=device)
    return(tmp)
  }

  gen_Gamma <- function(p, mask=FALSE){
      sigma = rinvgamma(n=K-1,15)
      Ip = diag(p)
      gvals = matrix(0,nr=K-1,nc=p)
      for(k in 1:(K-1)){
          gamma_k = rmvnorm(n=1, rep(0,p), sigma=sigma[k]*Ip)
          gvals[k,] = gamma_k
      }
      Gamma = torch_tensor(gvals, device=device)
      if(mask){
          Gamma[abs(Gamma) <0.1] = 0
          Gamma = Gamma*5
      }
      sigma = torch_tensor(sigma, device=device)
      return(list(sigma=sigma, Gamma = Gamma))
  }

  gen_X <- function(p, n_samples, add_const = TRUE){
    X = torch_randn(c(n_samples, p), device=device)
    if (add_const){
      X = torch_randn(c(n_samples, p-1), device=device)
      X = torch_cat(c(torch_ones(n_samples,1, device=device), X), dim=2)
    }
    return(X)
  }

  gen_theta <- function(K, D, mu, mvn=TRUE){
    if(mvn){
      theta = torch_zeros(K, D)
      mu = mu$cpu()
      A = distr_normal(0,0.2)$sample(c(K-1,K-1))$squeeze()
      Sigma = 2*torch_eye(K-1) + A$matmul(A$transpose(1,2))
      for (d in 1:D){
        eta_d = distr_multivariate_normal(mu[,d], Sigma)$sample()
        eta_d = torch_cat(c(eta_d, torch_tensor(0.0)), dim=1)
        theta[,d] = nnf_softmax(eta_d, dim=1)
      }
    } else{
        tmp = torch_cat(c(mu, torch_zeros(1, ncol(mu))), dim=1)
        theta = nnf_softmax(tmp, dim=1)
    }
    if(cuda_is_available()){
        theta = theta$cuda()
    }
    return(theta)
  }
  b_res = gen_b(K)
  bt = b_res$bt; br = b_res$br
  epi  = gen_epi(K)
  nuc = gen_nuc(K)
  clu = gen_clu(K)

  factors = list(bt = bt, br = br, epi = epi, nuc = nuc, clu = clu)

  T0 = gen_T(V,K, a = 300)

  cTrain = sample_count_from_nb(n = 20, mean = 150, n_sample = D)

  gres = gen_Gamma(p)
  gamma_sigma = gres$sigma; Gamma = gres$Gamma
  X = gen_X(p, n_samples= D,add_const = TRUE)
  mu = Gamma$matmul(X$transpose(1,2))

  theta = gen_theta(K=K, D = D, mu = mu)

  Ytrain = torch_zeros(c(3,3, 16, 4, 2, V, D), device=device)

  for(d in 1:length(cTrain)){
    m = gen_m(missing_prob = c(0.6, 0.1, 0.2, 0.1))
    
    zd = Categorical(cTrain[d], theta[,d])
    md = Categorical(cTrain[d], m)

    for (n in 1:length(zd)){
      t_dn = Categorical(torch_tensor(1), factors$bt[1:2,zd[n]])
      r_dn = Categorical(torch_tensor(1), factors$br[1:2,zd[n]])
      e_dn = Categorical(torch_tensor(1), factors$epi[,zd[n]])
      n_dn = Categorical(torch_tensor(1), factors$nuc[,zd[n]])
      c_dn = Categorical(torch_tensor(1), factors$clu[,zd[n]])

      v_dn = Categorical(torch_tensor(1), T0[t_dn, r_dn,,zd[n]])

      if (md[n] %in% c(3,4)){
        t_dn = 3
      }
      if (md[n] %in% c(2,4)){
        r_dn = 3
      }

      Ytrain[t_dn, r_dn, e_dn, n_dn, c_dn, v_dn, d] = Ytrain[t_dn, r_dn, e_dn, n_dn, c_dn, v_dn, d] + 1
    }
  }

  return(list(T0 = T0, Gamma=Gamma,factors = factors, theta=theta, count_matrix=Ytrain,X = X))

}

## Generate init parameters using NMF
# library(NMF)
# library(NNLM)





##########
## Trying to improve initialization

T0_NMF_init <- function(Y, K, max_iter, eps=1e-20){
  ## Initialize signature matrices using the results of nmf
  # suppressWarnings(nmf(as_array(tmp$cpu()),rank=K, model=list(H=E), seed="nndsvd", maxIter=10000L))
  if(cuda_is_available()){
    Y = Y$cpu()
  }
  Y_tmp = as_array(Y$sum(dim=c(1,2,3,4,5)))
  nmf_res0 = suppressWarnings(nmf(1.1*Y_tmp + 0.2, 
                              rank = K, seed="nndsvd", maxIter = max_iter))
  
  W_ = nmf_res0@fit@W
  H_ = nmf_res0@fit@H

  nmf_res1 = nmf_fit(mat = Y_tmp + 1e-14, h = H_, w = W_, K = K, max_iter = max_iter)
  W_ = nmf_res1@fit@W
  H_ = nmf_res1@fit@H


  theta = tryCatch({
      torch_tensor(H_/(colSums(H_)), device=device)},
      error = function(e) {
      torch_tensor(H_/(colSums(H_)+1e-10), device=device)}
  )

  # T0_CL, T0_CG, T0_TL, T0_TG fitting...
  X = as_array(Y[1,1]$sum(dim=c(1,2,3))) + 1e-2
  tmp = nmf_fit(X, H_, W_, K, max_iter)
  CL_init = tmp@fit@W 
  CL_init = CL_init/colSums(CL_init)

  X = as_array(Y[1,2]$sum(dim=c(1,2,3))) + 1e-2
  tmp = nmf_fit(X, H_, W_, K, max_iter)
  CG_init = tmp@fit@W 
  CG_init = CG_init/colSums(CG_init)

  X = as_array(Y[2,1]$sum(dim=c(1,2,3))) + 1e-2
  tmp = nmf_fit(X, H_, W_, K, max_iter)
  TL_init = tmp@fit@W 
  TL_init = TL_init/colSums(TL_init)

  X = as_array(Y[2,2]$sum(dim=c(1,2,3))) + 1e-2
  tmp = nmf_fit(X, H_, W_, K, max_iter)
  TG_init = tmp@fit@W 
  TG_init = TG_init/colSums(TG_init)

  # From arrays to torch tensors
  CL_init = torch_tensor(CL_init, device=device)
  CG_init = torch_tensor(CG_init, device=device)
  TL_init = torch_tensor(TL_init, device=device)
  TG_init = torch_tensor(TG_init, device=device)

  V = nrow(CL_init)

  T0 = torch_stack(c(CL_init, CG_init, TL_init, TG_init))$reshape(c(2,2,V,K))

  return(list(T0=T0, theta=theta))
}

factor_NMF_init <- function(Y,K,H, max_iter = 1000){
  factor_names = c("bt", "br","epi","nuc","clu")
  factor_dim = c(2,2,16,4,2)
  factors = list(bt = NULL, br = NULL, epi = NULL, nuc=NULL, clu=NULL)
  
  if (cuda_is_available()){
    Y = Y$cpu()
    H = H$cpu()
  }

  H = as_array(H)

  bt_init = torch_empty(factor_dim[1],K, device=device)
  for (i in 1:factor_dim[1]){
    Y_tmp = torch_sum(Y[i], dim=c(1,2,3,4)) + 1e-2
    tmp = nmf_fit(mat = as_array(Y_tmp), w = NULL, h= H, K=K,max_iter = max_iter)
    bt_init[i] = colSums(tmp@fit@W)
  }
  factors$bt = bt_init/torch_sum(bt_init, dim=1) 

  br_init = torch_empty(factor_dim[2],K, device=device)
  for (i in 1:factor_dim[2]){
    Y_tmp = torch_sum(Y[,i], dim=c(1,2,3,4)) + 1e-2
    tmp = nmf_fit(mat = as_array(Y_tmp), w = NULL, h= H, K=K,max_iter = max_iter)
    br_init[i] = colSums(tmp@fit@W)
  }
  factors$br = br_init/torch_sum(br_init, dim=1) 
  
  epi_init = torch_empty(factor_dim[3],K, device=device)
  for (i in 1:factor_dim[3]){
    Y_tmp = torch_sum(Y[,,i], dim=c(1,2,3,4)) + 1e-2
    tmp = nmf_fit(mat = as_array(Y_tmp), w = NULL, h= H, K=K,max_iter = max_iter)
    epi_init[i] = colSums(tmp@fit@W)
  }
  factors$epi = epi_init/torch_sum(epi_init, dim=1) 
  
  nuc_init = torch_empty(factor_dim[4],K, device=device)
  for (i in 1:factor_dim[4]){
    Y_tmp = torch_sum(Y[,,,i], dim=c(1,2,3,4)) + 1e-2
    tmp = nmf_fit(mat = as_array(Y_tmp), w = NULL, h= H, K=K,max_iter = max_iter)
    nuc_init[i] = colSums(tmp@fit@W)
  }
  factors$nuc = nuc_init/torch_sum(nuc_init,dim=1)
  
  clu_init = torch_empty(factor_dim[5],K, device=device)
  for (i in 1:factor_dim[5]){
    Y_tmp = torch_sum(Y[,,,,i], dim=c(1,2,3,4)) + 1e-2
    tmp = nmf_fit(mat = as_array(Y_tmp), w = NULL, h= H, K=K,max_iter = max_iter)
    clu_init[i] = colSums(tmp@fit@W)
  }
  factors$clu = clu_init/torch_sum(clu_init,dim=1) 

  return(factors)

}

nmf_fit <- function(mat, h, w, K, max_iter){
  if(is.null(w)){
    return(suppressWarnings(nmf(mat, rank=K, maxIter=max_iter, model=list(H=h),seed = "nndsvd")))
  }
  return(suppressWarnings(nmf(mat, rank=K, maxIter=max_iter, model=list(H = h, W= w), seed="none")))
}


#' Initialize parameters for variational EM algorithm
#'
#' @param Y count tensor
#' @param X matrix of covariates
#' @param K number of signatures or latent factors
#' @param max_iter maximum number of iterations
#' @return list of torch tensors for input into variational EM
#' @export
NMFinit <- function(Y, X, K, max_iter){
  
  t0_res = T0_NMF_init(Y, K, max_iter)
  theta = t0_res$theta
  T0 = t0_res$T0

  factors = factor_NMF_init(Y,K, theta, max_iter=10000)

  if (!is.null(X)){
    p = ncol(X)
    D = nrow(X)
    eta = logit_op(theta)
    Xi = eta$matmul(torch_pinverse(X$transpose(1,2)))
    tmp = 0.1*torch_rand(K-1, p, 2, device=device)
    zeta = torch_eye(p, device=device) + tmp$matmul(tmp$transpose(-1,-2))
    gamma_sigma = torch_ones(K-1, device=device)
    Sigma = torch_eye(K-1, device=device)*5
    Delta = Sigma$`repeat`(c(D,1,1))
    return(list( covs= factors, eta = eta, Delta = Delta, Xi = Xi, T0 = T0, Sigma=Sigma, gamma_sigma = gamma_sigma, zeta = zeta))
  } else {
    eta = theta
    H = theta$mean(dim=2)
    return(list(eta = eta, H=H))
  }

}

#' Initialize using random intiial values
#'
#' @param Y count tensor  
#' @param X tensor of covariates (a p x D matrix)
#' @param anno_dims list of dimensions of annotations (default = list(epi_dim = 16, nuc_dim = 4, clu_dim = 2, V= 96))
#' @param K number of signatures or latent factors
#' @export
random_init <- function(Y,X, anno_dims = list(epi_dim = 16, nuc_dim = 4, clu_dim = 2, V= 96),K){
  D = Y$shape[length(dim(Y))]
  cl_  = t(rdirichlet(K,rep(1,anno_dims$V)))
  cl_ = torch_tensor(cl_,device=device)
  cg_  = t(rdirichlet(K,rep(1,anno_dims$V)))
  cg_ = torch_tensor(cg_,device=device)
  tl_  = t(rdirichlet(K,rep(1,anno_dims$V)))
  tl_ = torch_tensor(tl_,device=device)
  tg_  = t(rdirichlet(K,rep(1,anno_dims$V)))
  tg_ = torch_tensor(tg_,device=device)

  T0  = torch_stack(c(cl_, cg_, tl_, tg_))$reshape(c(2,2,anno_dims$V,K))

  theta = torch_tensor(t(rdirichlet(D,rep(1,K))),device=device)
  eta = logit_op(theta)

  bt_ = torch_tensor(t(rdirichlet(K,rep(1,2))),device=device)
  br_ = torch_tensor(t(rdirichlet(K,rep(1,2))), device=device)
  epi_ = torch_tensor(t(rdirichlet(K,rep(1,anno_dims$epi_dim))), device=device)
  nuc_ = torch_tensor(t(rdirichlet(K,rep(1,anno_dims$nuc_dim))), device=device)
  clu_ = torch_tensor(t(rdirichlet(K,rep(1,anno_dims$clu_dim))), device=device)

  factors = list(bt = bt_, br = br_, epi = epi_, nuc = nuc_, clu = clu_)
  if(!is.null(X)){
    p = ncol(X)
    Xi = eta$matmul(torch_pinverse(X$transpose(1,2)))
    tmp = 0.1*torch_rand(K-1, p, 2, device=device)
    zeta = torch_eye(p, device=device) + tmp$matmul(tmp$transpose(-1,-2))
    gamma_sigma = torch_ones(K-1, device=device)
    Sigma = torch_eye(K-1, device=device)*5
    Delta = Sigma$`repeat`(c(D,1,1))
  } else{
    message("Please provide non-null design tensor X")
  }

  return(list( covs= factors, eta = eta, Delta = Delta, Xi = Xi, T0 = T0, Sigma=Sigma, gamma_sigma = gamma_sigma, zeta = zeta))

}


## Function to generate count_matrix from a fitted model
init_from_mod <- function(mod0, Y){
  ## Input:
  ##    - mod0: HDsig output from runEM
  ##    - Y: original count tensor to get missing rate and total mutation number. 
  ## Output:
  ##    - Ytain, a 3,3,16,4,2,V,D torch tensor of counts. 
  cTrain = Y$sum(dim=c(1,2,3,4,5,6))
  V = 96
  D = length(cTrain)
  Ytrain = torch_zeros(c(3,3, 16, 4, 2, V, D), device=device)
  
  factors = mod0$Bparam$factors
  theta = nnf_softmax(torch_cat(c(mod0$VIparam$lambda$clone(), torch_zeros(1, D , device = device)), dim=1), dim=1)
  T0 = mod0$Bparam$T0$clone()

  m = make_m__(Y)

  for (d in 1:length(cTrain)){
    zd = Categorical(cTrain[d], theta[,d])
    md  = Categorical(cTrain[d],m)

    for(n in 1:length(zd)){
      t_dn = Categorical(torch_tensor(1), factors$bt[1:2,zd[n]])
      r_dn = Categorical(torch_tensor(1), factors$br[1:2,zd[n]])
      n_dn = Categorical(torch_tensor(1), factors$nuc[,zd[n]])
      e_dn = Categorical(torch_tensor(1), factors$epi[,zd[n]])
      c_dn = Categorical(torch_tensor(1), factors$clu[,zd[n]])

      v_dn = Categorical(torch_tensor(1), T0[t_dn, r_dn,,zd[n]])

      if (md[n]$item() %in% c(3,4)){
        t_dn = 2
      }
      if (md[n]$item() %in% c(2,4)){
        r_dn = 2
      }

      Ytrain[t_dn, r_dn, e_dn, n_dn, c_dn, v_dn, d] = Ytrain[t_dn, r_dn, e_dn, n_dn, c_dn, v_dn, d] + 1
    }
  }

  return(Ytrain)
}



  