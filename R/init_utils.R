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
  tmp = torch_tensor(tmp, dtype = torch_long(), device=device)
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
    bt = t(rdirichlet(K, c(0.5, 0.5)))
    bt = torch_tensor(bt, device=device)

    br = t(rdirichlet(K, c(0.5,0.5)))
    br = torch_tensor(br, device=device)
    return(list(bt=bt, br = br))
  }

  ## generate k_epi
  gen_epi <- function(K, a = 10){
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
    tmp = torch_tensor(rnbinom(size=n, n = n_sample, mu=mean), device=device)
    return(tmp)
  }

  gen_Gamma <- function(p){
      sigma = rinvgamma(n=(K-1),p)
      Ip = diag(p)
      gvals = rmvnorm(n=K-1, rep(0,p), sigma=sigma*Ip)
      Gamma = torch_tensor(gvals, device=device)
      #Gamma[abs(Gamma)< 0.2] = 0
      #Gamma = Gamma * 3
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

  gen_theta <- function(K, D, mu){
    tmp = torch_cat(c(mu, torch_zeros(1, ncol(mu), device=device)), dim=1)
    theta = nnf_softmax(tmp, dim=1)
    return(theta)
  }

  b_res = gen_b(K)
  bt = b_res$bt; br = b_res$br
  epi  = gen_epi(K)
  nuc = gen_nuc(K)
  clu = gen_clu(K)

  factors = list(bt = bt, br = br, epi = epi, nuc = nuc, clu = clu)

  T0 = gen_T(V,K, a = K)

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
      t_dn = Categorical(torch_tensor(1), factors$bt[,zd[n]])
      r_dn = Categorical(torch_tensor(1), factors$br[,zd[n]])
      e_dn = Categorical(torch_tensor(1), factors$epi[,zd[n]])
      n_dn = Categorical(torch_tensor(1), factors$nuc[,zd[n]])
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

  return(list(T0 = T0, Gamma=Gamma,factors = factors, theta=theta, count_matrix=Ytrain,X = X))

}

## Generate init parameters using NMF
# library(NMF)
# library(NNLM)


covs_to_F <- function(bt,br, epi, nuc, clu){
  K = ncol(bt)

  F_tensor = torch_ones(c(3,3,16,4,2,K), device=device)
  for (i in 1:3) {F_tensor[i] = F_tensor[i] * bt[i]}
  for (i in 1:3) {F_tensor[,i] = F_tensor[,i]*br[i]}
  for (i in 1:6) {F_tensor[,,i] = F_tensor[,,i]*epi[i]}
  for (i in 1:4) {F_tensor[,,,i] = F_tensor[,,,i]*nuc[i]}
  for (i in 1:2) {F_tensor[,,,,i] = F_tensor[,,,,i]*clu[i]}
  
  return(F_tensor)

}

T0toT <- function(T0){
  CL = T0[1,1] ; CG = T0[1,2]; TL = T0[2,1]; TG = T0[2,2]

  V = nrow(CL)
  K = ncol(CL)

  T0_C_ = (CL + CG)/2
  T0_T_ = (TL + TG)/2
  T0_L = (CL + TL)/2
  T0_G = (CG + TG)/2
  T0__ = (T0_L + T0_G)/2

  T_tensor = torch_empty(c(3,3,16,4,2,V,K), device=device)
  T_tensor[1,1] = CL
  T_tensor[1,2] = T0_C_
  T_tensor[1,3] = CG
  T_tensor[2,1] = T0_L
  T_tensor[2,2] = T0__
  T_tensor[2,3] = T0_G
  T_tensor[3,1] = TL
  T_tensor[3,2] = T0_T_
  T_tensor[3,3] = TG

  return(T_tensor)
}

stack <- function(T0, bt, br, n_epi=16, n_nuc=4, n_clu=2){ 

  ## Input:
  ##   T0, torch_tensor: the signature matrices (size: 2x2x96xD). D= number of samples
  ##   bt, torch_tensor: the transcriptions bias size 2xK
  ##   br, torch_tensor: the replication bias size 2xK
  ## Output:
  ##    T tensor (size: 3x3x16x4x2)

  CL = T0[1,1] ; CG = T0[1,2]; TL = T0[2,1]; TG = T0[2,2]

  V = nrow(CL)
  K = ncol(CL)

  T_tensor = torch_empty(c(3,3,n_epi,n_nuc,n_clu,V,K), device=device)

  bt0 = bt[1] # first row of bt (size : 1 x K)
  br0 = br[1] # first row of br (size : 1 x K)

  T0_C_ = br0*CL + (1-br0)*CG
  T0_T_ = br0*TL + (1-br0)*TG
  T0_L  = bt0*CL + (1-bt0)*TL
  T0_G  = bt0*CG + (1-bt0)*TG

  T0__  = bt0*br0*CL + bt0*(1-br0)*CG + (1-bt0)*br0*TL + (1-bt0)*(1-br0)*TG

  T_tensor[1,1] = CL ; T_tensor[2,1] = T0_L ; T_tensor[3,1] = TL
  T_tensor[1,2] = T0_C_ ; T_tensor[2,2] = T0__ ; T_tensor[3,2] = T0_T_
  T_tensor[1,3] = CG ; T_tensor[2,3] = T0_G ; T_tensor[3,3] = TG

  return(T_tensor)
}

factors_to_F <- function(factors, 
                         factor_dim = c(2,2,16,4,2),
                         missing_rate = NULL){

    # K : number of signatures
    # n_epi : number of categories in epi
    # n_nuc : number of categories in nuc
    # n_clu : number of categories in clu
    # factors: a list with br, bt, epi, nuc, clu
    # returns F_tensor which is a 3x3x16x4x2x96 tensor

    epi = factors$epi; nuc = factors$nuc; clu = factors$clu
    bt = factors$bt; br = factors$br
    
    K = ncol(epi)
    n_epi = factor_dim[3]
    n_nuc = factor_dim[4]
    n_clu = factor_dim[5]

    F_tensor = torch_ones(c(3,3,n_epi,n_nuc,n_clu,K), device=device)

    for (l in 1 : n_epi) {F_tensor[,,l] = F_tensor[,,l]*epi[l]}
    for (l in 1 : n_nuc) {F_tensor[,,,l] = F_tensor[,,,l]*nuc[l]}
    for (l in 1 : n_clu) {F_tensor[,,,,l] = F_tensor[,,,,l]*clu[l]}

    if (!is.null(missing_rate)){ 
        for (i in 1 : 2) {
            for (j in 1 : 2) {
                F_tensor[i, j] = F_tensor[i, j]$clone() * bt[i] * br[j] * missing_rate[1,1]
            } 
            F_tensor[i, 3] = F_tensor[i,3]$clone() * bt[i] * missing_rate[1, 2]
        }
        for (j in 1 : 2){
            F_tensor[3, j] = F_tensor[3,j]$clone() * br[j] * missing_rate[2, 1]
        }
        F_tensor[3,3] = F_tensor[3,3]$clone() *  missing_rate[2,2]
    } else{
        for (i in 1 : 3){
            F_tensor[i] = F_tensor[i]$clone() * bt[i]
        }
        for (i in 1 : 3){
            F_tensor[,i] = F_tensor[,i]$clone()  * F_tensor[,i] * br[i]
        }
    }

    return(F_tensor)
}


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

  bt_init = torch_empty(3,K, device=device)
  for (i in 1:3){
    Y_tmp = torch_sum(Y[i], dim=c(1,2,3,4)) + 1e-2
    tmp = nmf_fit(mat = as_array(Y_tmp), w = NULL, h= H, K=K,max_iter = max_iter)
    bt_init[i] = colSums(tmp@fit@W)
  }
  factors$bt = bt_init/torch_sum(bt_init, dim=1) 

  br_init = torch_empty(3,K, device=device)
  for (i in 1:3){
    Y_tmp = torch_sum(Y[,i], dim=c(1,2,3,4)) + 1e-2
    tmp = nmf_fit(mat = as_array(Y_tmp), w = NULL, h= H, K=K,max_iter = max_iter)
    br_init[i] = colSums(tmp@fit@W)
  }
  factors$br = br_init/torch_sum(br_init, dim=1) 
  
  epi_init = torch_empty(16,K, device=device)
  for (i in 1:16){
    Y_tmp = torch_sum(Y[,,i], dim=c(1,2,3,4)) + 1e-2
    tmp = nmf_fit(mat = as_array(Y_tmp), w = NULL, h= H, K=K,max_iter = max_iter)
    epi_init[i] = colSums(tmp@fit@W)
  }
  factors$epi = epi_init/torch_sum(epi_init, dim=1) 
  
  nuc_init = torch_empty(4,K, device=device)
  for (i in 1:4){
    Y_tmp = torch_sum(Y[,,,i], dim=c(1,2,3,4)) + 1e-2
    tmp = nmf_fit(mat = as_array(Y_tmp), w = NULL, h= H, K=K,max_iter = max_iter)
    nuc_init[i] = colSums(tmp@fit@W)
  }
  factors$nuc = nuc_init/torch_sum(nuc_init,dim=1)
  
  clu_init = torch_empty(2,K, device=device)
  for (i in 1:2){
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
  