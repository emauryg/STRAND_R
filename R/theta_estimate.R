# ## Functions to estimate theta


## Calculate our own gradient descent

update_eta_Delta <- function(T0, covs, eta, Sigma, Y,Xi, X, hyp){
  SigmaInv = Sigma$inverse()
  mu = Xi$matmul(X$transpose(1,2))
  lr = hyp$lr
  max_iter = hyp$max_iter
  tol = hyp$tol

  D = ncol(eta)
  Delta = torch_empty(c(D,K-1, K-1), device=device)
  ## Batch size used from suggestions here:
  ## https://machinelearningmastery.com/gentle-introduction-mini-batch-gradient-descent-configure-batch-size/
  batches = msplit(1:D, ceiling(D/128))
  for (b in batches){
      yphi_ = yphi(covs=covs, T0 = T0, Y= Y[..,b,drop=FALSE], 
                  missing_rate = make_m__(Y[..,b,drop=FALSE]), X = X[b,,drop=FALSE], context=TRUE,eta = eta[,b,drop=FALSE])
      lp = Laplace_fit(eta[,b,drop=FALSE], mu[,b,drop=FALSE],yphi_,SigmaInv,max_iter, tol, lr)
      Delta[b] = lp$Delta
      eta[,b] = lp$eta
  }
  return(list(eta=eta, Delta = Delta))
}

Laplace_fit <- function(eta, mu, yphi_,SigmaInv, max_iter, tol, lr){
  old_grad = 1e10

  # Initialize 1st and 2nd moment values
  s = 0
  r = 0
  for (it in 1:max_iter){
    g = grad_func(eta, mu, yphi_, SigmaInv)
    res_optim = adam_optim0(eta,s,r, lr, g,it)
    eta = res_optim$eta
    s = res_optim$s
    r = res_optim$r
    g_mean = g$sum(dim=1)$sum()/ncol(eta)
    if(stop_theta(old_grad,g_mean$item(),tol)){
      break
    } else {
      old_grad = g_mean$item()
    }
  }
    
  nu = calc_hessInv(eta, yphi_,SigmaInv)
  return(list(eta=eta, Delta=nu))

}

grad_func <- function(eta,mu,yphi_,SigmaInv){
  # Calculate the gradient
  d = ncol(eta)
  lam0 = torch_cat(c(eta, torch_zeros(1,d, device = device)), dim=1)
  grad = torch_mm(SigmaInv, eta-mu) - yphi_[..,1:-2]$transpose(1,2) + 
    yphi_$sum(dim=2,keepdim=TRUE)$transpose(1,2)*nnf_softmax(lam0, dim=1)[1:-2]
  return(grad)
}

calc_hessInv <- function(eta, yphi_, SigmaInv){
  Yn = yphi_$sum(dim =2, keepdim=TRUE)
  d = ncol(eta)
  nu = torch_empty(c(d,nrow(eta), nrow(eta)), device=device)
  for(i in 1:d){
      eta_d = torch_cat(c(eta[,i, drop=FALSE], torch_zeros(1,device=device)), dim=1)
      theta = nnf_softmax(eta_d, dim=1)[1:-2]$reshape(c(-1,1))
      hess = SigmaInv - Yn[i,]*(theta$matmul(theta$transpose(1,2)) - torch_diag_embed(theta$squeeze()))
      nu[i] = hess$inverse()
  }
  return(nu)
}

adam_optim0 <- function(eta,s,r, lr,grad, it, rho1=0.9, rho2=0.999, delta = 1e-10){
  # Update the parameters
  s  = rho1*s + (1-rho1)*grad
  r  = rho2*r + (1-rho2)*grad$pow(2)

  s_hat = s/(1-rho1^it)
  r_hat = r/(1-rho2^it)
  delta = -lr*s_hat/(sqrt(r_hat) + delta)
  eta = eta + delta
  return(list(eta=eta, s=s, r=r))
}

stop_theta <- function(old_grad, grad, tol){
  rat_cri = FALSE
  abs_cri = FALSE
  if(abs(grad - old_grad)/ (abs(old_grad) + 1e-10) < tol$ratio){
    rat_cri = TRUE
  }

  if (abs(grad - old_grad) < tol$abs){
    abs_cri = TRUE
  }
 
  if(rat_cri & abs_cri) {return (TRUE)} else{ return(FALSE)}
}