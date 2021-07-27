## Functions and scripts to run TensorSignature

# anno_dims = list(epi_dim = 16, nuc_dim = 4, clu_dim = 2, V= 96)

#' Initialize TensorSignature algorithm
#'
#' @param anno_dims Annotation dimensions
#' @param Y count tensor
#' @param X design matrix
#' @param K Number of clusters
#' @param max_iter Number of iterations
#' @export
init_tensorsig <- function(Y,X,K, max_iter, anno_dims){
  ## Initialize algorithm for tensorsignature

  init_tmp = NMFinit(Y,X,K,max_iter)

  T0_ = init_tmp$T0$clone()
  
  cl_ = logit_op(T0_[1,1])
  cg_ = logit_op(T0_[1,2])
  tl_ = logit_op(T0_[2,1])
  tg_ = logit_op(T0_[2,2])

  T0_ = torch_stack(c(cl_,cg_, tl_, tg_))$reshape(c(2,2,anno_dims$V-1,K))

  bt_ = torch_randn(K,device=device)
  br_ = torch_randn(K,device=device)
  at_ = torch_randn(K,device=device)
  ar_ = torch_randn(K,device=device)

  epi_ = torch_randn(c(anno_dims$epi_dim - 1, K), device=device)
  nuc_ = torch_randn(c(anno_dims$nuc_dim -1, K), device=device)
  clu_ = torch_randn(c(anno_dims$clu_dim - 1, K), device=device)

  E_ = torch_randn(c(K, nrow(X)), device=device)

  return(list(T0_ = T0_, bt_ = bt_, br_ = br_, at_ = at_, ar_ = ar_, epi_=epi_, nuc_=nuc_, clu_=clu_, E_=E_))
}

stop_ts <- function(old_loss,new_loss){
  if(abs(new_loss - old_loss)/abs(old_loss) < 0.001){
      return(TRUE)
  } else {
      return(FALSE)
  }
}


run_TS <- torch::nn_module(
  classname = "run_TS",
  initialize = function(init_list){
    self$bt_ = nn_parameter(init_list$bt_)
    self$br_ = nn_parameter(init_list$br_)
    self$ar_ = nn_parameter(init_list$ar_)
    self$at_ = nn_parameter(init_list$at_)
    self$epi_ = nn_parameter(init_list$epi_)
    self$nuc_ = nn_parameter(init_list$nuc_)
    self$clu_ = nn_parameter(init_list$clu_)
    self$T0_ = nn_parameter(init_list$T0_)
    self$E_ = nn_parameter(init_list$E_)
  }, 
  forward = function(Y, tau, anno_dims){
    E = torch_exp(self$E_)
    V = anno_dims$V
    K = length(self$bt_)

    ## Calculate T1
    cl_ = self$T0_[1,1]
    cg_ = self$T0_[1,2]
    tl_ = self$T0_[2,1]
    tg_ = self$T0_[2,2]
  
    cl = nnf_softmax(torch_cat(c(cl_, torch_zeros(1, K, device=device)), dim=1), dim=1)
    cg = nnf_softmax(torch_cat(c(cg_, torch_zeros(1, K, device=device)), dim=1), dim=1)
    tl = nnf_softmax(torch_cat(c(tl_, torch_zeros(1, K, device=device)), dim=1), dim=1)
    tg = nnf_softmax(torch_cat(c(tg_, torch_zeros(1, K, device=device)), dim=1), dim=1)

    # cl = self$T0_[1,1]
    # cg = self$T0_[1,2]
    # tl = self$T0_[2,1]
    # tg = self$T0_[2,2]

    c_ = 0.5*cl + 0.5*cg
    t_ = 0.5*tl + 0.5*tg
    l_ = 0.5*cl + 0.5*tl
    g_ = 0.5*cg  + 0.5*tg
    to__ = (cl+cg+tl +tg)/4
    T1 = torch_empty(c(3, 3, V, K), device=device)
    T1[1,1] = cl; T1[2,1] = l_; T1[3,1] = tl;
    T1[1,2] = c_; T1[2,2] = to__; T1[3,2] = t_;
    T1[1,3] = cg; T1[2,3] = g_ ; T1[3,3] = tg;
      
    ## Calculate B
    B = torch_stack(c(self$bt_ + self$br_, self$bt_-self$br_, self$bt_,
                    -self$bt_ + self$br_, -self$bt_ - self$br_, -self$bt_,
                    self$br_, -self$br_, torch_zeros(K, device=device)))$reshape(c(3,3,1,K))
    B = torch_exp(B)
      
      

    ## Calculate A
    A = torch_stack(c(self$at_ + self$ar_, self$at_ - self$ar_, self$at_,
                  self$at_ + self$ar_, -self$at_ - self$ar_, self$at_, 
                  self$ar_, self$ar_, torch_zeros(K, device=device)))$reshape(c(3,3,1,K))
    A = torch_exp(A)

    ## Calculate K_epi, K_nuc, and K_clu
    K_epi = torch_exp(torch_cat( c(self$epi_, torch_zeros(1,K, device=device)), dim=1))
    K_nuc = torch_exp(torch_cat( c(self$nuc_, torch_zeros(1,K, device=device)), dim=1))
    K_clu = torch_exp(torch_cat( c(self$clu_, torch_zeros(1,K, device=device)), dim=1))
    

    ## Calculate T_strand
    T_strand = T1 * B * A

    ## Calculate K_tensor
    K_tensor = K_epi$view(c(1,1,-1,1,1,K)) * K_nuc$view(c(1,1,1,-1,1,K)) *
      K_clu$view(c(1,1,1,1,-1, K))

    ## Calculate T_tensor
    T_tensor = T_strand$view(c(3,3,1,1,1,-1,K))*K_tensor$unsqueeze(-2)
    
    ## Calculate Chat
    Chat = T_tensor$matmul(E)
      
    ## Loss 
    D = ncol(E)
    
    L = (-Y*torch_log(Chat) + (Y + tau)*torch_log(Chat + tau))$sum()/D
    return(L)
  }
)


#' Fit TensorSignature model
#'
#' @param Y count tensor
#' @param X design matrix
#' @param anno_dims annotation dimensions
#' @param K number of signatures
#' @param tau regularization parameter (default: 50)
#' @param lr learning rate (default: 0.01)
#' @param max_iter maximum number of iterations (default: 1000)
#' @export
fit_TS <- function(Y,X,K, anno_dims, lr=0.01, tau=50, max_iter = 1000){
  converged = FALSE
  old_loss = 1e10
  it = 0

  init_list = init_tensorsig(Y,X,K, max_iter = max_iter, anno_dims = anno_dims)
  tmp_mod = run_TS(init_list)
  optimizer = optim_adam(tmp_mod$parameters, lr=lr)

  start = Sys.time()
  while( converged == FALSE && it <= max_iter){
    it= it +1
    if (it %% 100 ==0){
      message(paste("Current iteration:",it, ", loss =",old_loss))
    }
    optimizer$zero_grad()
    new_loss = tmp_mod(Y, tau=tau, anno_dims = anno_dims)
    new_loss$backward()
    optimizer$step()
    converged = stop_ts(old_loss, new_loss$item())
    old_loss = new_loss$item()

  }

  message(paste("It took:",
    difftime(Sys.time(),start, units= "mins"), " to converge."))

    return(get_output(tmp_mod))

}

get_output <- function(tmp_mod){
  
  bt = tmp_mod$parameters$bt_$detach()$clone()
  br = tmp_mod$parameters$br_$detach()$clone()
  at = tmp_mod$parameters$at_$detach()$clone()
  ar = tmp_mod$parameters$ar_$detach()$clone()
  epi = torch_exp(torch_cat(c(tmp_mod$parameters$epi_$detach()$clone(), torch_zeros(1,K, device=device)),dim=1))
  nuc = torch_exp(torch_cat(c(tmp_mod$parameters$nuc_$detach()$clone(), torch_zeros(1,K, device=device)),dim=1))
  clu = torch_exp(torch_cat(c(tmp_mod$parameters$clu_$detach()$clone(), torch_zeros(1,K, device=device)),dim=1))
  cl = nnf_softmax(torch_cat(c(tmp_mod$parameters$T0_[1,1]$detach()$clone(), torch_zeros(1, K, device=device)), dim=1), dim=1)
  cg = nnf_softmax(torch_cat(c(tmp_mod$parameters$T0_[1,2]$detach()$clone(), torch_zeros(1, K, device=device)), dim=1), dim=1)
  tl = nnf_softmax(torch_cat(c(tmp_mod$parameters$T0_[2,1]$detach()$clone(), torch_zeros(1, K, device=device)), dim=1), dim=1)
  tg = nnf_softmax(torch_cat(c(tmp_mod$parameters$T0_[2,2]$detach()$clone(), torch_zeros(1, K, device=device)), dim=1), dim=1)
  V= nrow(cl)
  T1 = torch_empty(c(2, 2, V, K), device=device)
  T1[1,1] = cl
  T1[1,2] = cg
  T1[2,1] = tl
  T1[2,2] = tg


  E = torch_exp(tmp_mod$parameters$E_$detach()$clone())

  Bparam = list(T0 = T1, factors = list(br =br,bt=bt, at=at, ar=ar, epi=epi, nuc=nuc, clu=clu))
  VIparam = list(E = E)

  return(list(VIparam = VIparam, Bparam=Bparam))

}

#' Calculate T tensor from TensorSignature model
#'
#' @param tmp_mod TensorSignature model
#' @export
calculate_T_ts <- function(tmp_mod){
  bt = tmp_mod$Bparam$factors$bt$clone()
  br = tmp_mod$Bparam$factors$br$clone()
  at = tmp_mod$Bparam$factors$at$clone()
  ar = tmp_mod$Bparam$factors$ar$clone()

  cl = tmp_mod$Bparam$T0[1,1]$clone()
  cg = tmp_mod$Bparam$T0[1,1]$clone()
  tl = tmp_mod$Bparam$T0[2,1]$clone()
  tg = tmp_mod$Bparam$T0[2,2]$clone()

  c_ = 0.5*cl + 0.5*cg
  t_ = 0.5*tl + 0.5*tg
  l_ = 0.5*cl + 0.5*tl
  g_ = 0.5*cg  + 0.5*tg
  to__ = (cl+cg+tl +tg)/4
  V= nrow(cl)
  T1 = torch_empty(c(3, 3, V, K), device=device)
  T1[1,1] = cl; T1[2,1] = l_; T1[3,1] = tl;
  T1[1,2] = c_; T1[2,2] = to__; T1[3,2] = t_;
  T1[1,3] = cg; T1[2,3] = g_ ; T1[3,3] = tg;

  B = torch_stack(c(bt + br, bt-br, bt,
                -bt + br, -bt - br, -bt,
                br, -br, torch_zeros(K, device=device)))$reshape(c(3,3,1,K))
  B = torch_exp(B)

  A = torch_stack(c(at + ar, at - ar, at,
              at + ar, -at - ar, at, 
              ar, ar, torch_zeros(K, device=device)))$reshape(c(3,3,1,K))
  A = torch_exp(A)

  K_epi = tmp_mod$Bparam$factors$epi$clone()
  K_nuc = tmp_mod$Bparam$factors$nuc$clone()
  K_clu = tmp_mod$Bparam$factors$clu$clone()

  T_strand = T1 * B * A

  K_tensor = K_epi$view(c(1,1,-1,1,1,K)) * K_nuc$view(c(1,1,1,-1,1,K)) *
  K_clu$view(c(1,1,1,1,-1, K))
  T_tensor = T_strand$view(c(3,3,1,1,1,-1,K))*K_tensor$unsqueeze(-2)

  return(T_tensor)

}


transform_E <- function(mod_ts,Y,K, tau=50,lr=0.001, max_iter=10000){
  converged = FALSE
  old_loss = 1e10
  it = 0
  K = K
  D = Y$size(dim=-1)

  run_E = nn_module(
    classname= "run_E",
    initialize = function(E0){
      self$E_ = nn_parameter(E0)
    },
    forward = function(Y,T_tensor, tau){
      exp_E = torch_exp(self$E_)
      Chat = T_tensor$matmul(exp_E)
      D = ncol(exp_E)
      L = (-Y*torch_log(Chat)+ (Y+tau)*torch_log(Chat+tau))$sum()/D
      return(L)
    }
  )
  T_tensor = calculate_T_ts(mod_ts)$clone()
  E0 = torch_randn(c(K, D), device=device)
  tmp_E = run_E(E0)
  optimizer = optim_sgd(tmp_E$parameters,lr=lr)
  start = Sys.time()
  while(converged==FALSE && it<= max_iter){
    it = it + 1
    if(it %% 100 == 0){
      message(paste("Current iteration:", it,", loss =", round(old_loss,2)))
    }
    optimizer$zero_grad()
    new_loss = tmp_E(Y,T_tensor, tau=tau)
    new_loss$backward()
    optimizer$step()
    converged = stop_ts(old_loss, new_loss$item())
    old_loss = new_loss$item()
  }

  E = torch_exp(tmp_E$parameters$E_$detach())
  message(paste("It took:", difftime(Sys.time(),start, unit="mins",
    " mins to converge.")))
  return(E)

}