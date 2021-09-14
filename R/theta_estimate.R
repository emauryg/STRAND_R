# ## Functions to estimate theta

# ## Lambda estimation without z collapsing
# #library(batch)
# estimate_theta <- torch::nn_module(  
#   classname = "estimate_theta",
#   initialize = function(eta){
#     self$eta = nn_parameter(eta)
#     #self$mu = mu
#     # self$D = ncol(self$eta)
#   },
#   forward = function(yphi_, Sigma, mu, by_batch = TRUE, T0,covs,Y,X){
#     # SigmaInv = Sigma$inverse()
#     if(by_batch){
#       D = ncol(self$eta)
#       ## This is an option to calculate the loss function that is slower, but more memory efficient. 
#       ## When D > 100 this option is recommended. 
#       ## Ideadly we could change the step sizes by chunks of at most 50. 
#       batches = msplit(1:ncol(self$eta), ceiling(ncol(self$eta)/1))
#       fun2 = 0
#       SigmaInv = Sigma$inverse()
#       for (b in batches){
#         gc()
#         yphi_ = yphi(covs=covs, T0 = T0, Y= Y[..,b,drop=FALSE], missing_rate = make_m__(Y[..,b,drop=FALSE]), X = X[b,,drop=FALSE], context=TRUE,eta = self$eta[,b,drop=FALSE])
#         if(length(b) == 1){
#           theta = nnf_softmax(torch_cat(c(self$eta[,b, drop=FALSE], torch_zeros(1,1, device = device)), dim=1), dim=1)
#           diff1 = self$eta[,b, drop=FALSE]- mu[,b, drop=FALSE]
#           fun1 = -0.5*diff1$transpose(1,2)$matmul(SigmaInv)$matmul(diff1) + yphi_$matmul(torch_log(theta+1e-14))$sum()
#           fun2 =  fun2 - fun1
#           rm(theta); rm(diff1); rm(fun1); rm(yphi_)
#           gc()
#         } else {
#           theta = nnf_softmax(torch_cat(c(self$eta[,b,drop=FALSE], torch_zeros(1, length(b), device = device)), dim=1), dim=1)
#           diff1 = self$eta[,b, drop=FALSE] - mu[,b, drop=FALSE]
#           fun1 = torch_diag(-0.5*diff1$transpose(1,2)$matmul(SigmaInv)$matmul(diff1)) + torch_diag(yphi_$matmul(torch_log(theta+1e-14))$sum(dim=c(1,2,3,4,5,7)))
#           fun2 = fun2 -fun1$mean()
#           rm(theta); rm(diff1); rm(fun1); rm(yphi_)
#           gc()
#         }
#       }
#     } else {
#       theta = nnf_softmax(torch_cat(c(self$eta, torch_zeros(1, D, device = device)), dim=1), dim=1)
#       diff1 = self$eta - self$mu
#       fun = torch_diag(-0.5*diff1$transpose(1,2)$matmul(SigmaInv)$matmul(diff1))
#       fun1 = fun + torch_diag(yphi_$matmul(torch_log(theta + 1e-14))$sum(dim=c(1,2,3,4,5,7)))
#       fun2 = -fun1$mean()
#     }
#     return(fun2)
#   } 
# )



# update_eta_Delta <- function(T0, covs, eta, Sigma, Y,Xi, X, hyp){
#   lr = hyp$lr 
#   max_iter = hyp$max_iter
#   tol = hyp$tol
#   K = nrow(Sigma) + 1
#   D = ncol(eta)
#   Delta = torch_empty(c(D,K-1, K-1), device=device)
#   mu = Xi$matmul(X$transpose(1,2))
#   old_loss = 1e10
#   it=0
#   converged = FALSE
  
#   yphi_ = 0
#   tmp_mod = estimate_theta(eta)
#   optimizer = optim_adam(tmp_mod$parameters, lr = lr)
#   while (converged == FALSE && it <= max_iter){
#     it = it+1
#     optimizer$zero_grad()
#     new_loss = tmp_mod(yphi_, Sigma, mu,by_batch=TRUE, T0, covs,Y,X)
#    # current implementation is memory intensive, need to call gc()
#     new_loss$backward()
#     optimizer$step() 
#     converged = theta_stop(new_loss$item(), old_loss, tol)
#     old_loss = new_loss$item()
#     gc()
#   }
#   eta= tmp_mod$parameters$eta$detach()
#   SigInv = Sigma$inverse()
#   TF = tf(T0, covs, make_m__(Y))
#   for(d in 1:D){
#     Y_d = Y[,,,,,,d]
#     Delta[d] = calc_hessInv(eta[,d], TF, Y_d, SigInv)
#   }
#   return(list(eta=eta, Delta = Delta))
# }


  
# theta_stop <- function(loss, old_loss, tol){
#   rat_cri = FALSE; abs_cri = FALSE
#   if(abs(loss - old_loss)/abs(old_loss) < tol$ratio){
#     rat_cri = TRUE
#   }
#   if (abs(loss - old_loss) < tol$abs){
#     abs_cri = TRUE
#   }
#   if(rat_cri & abs_cri) {return (TRUE)} else{ return(FALSE)}
# }

# calc_hessInv <- function(eta_d, T_tensor, Y_d, Sigma_inv){ 
#     x = eta_d
#     eta_d = torch_cat(c(x, torch_zeros(1, device=device)), dim=1)
#     theta_d = nnf_softmax(eta_d, dim=1)[1:-2]$reshape(c(-1,1))
    
#     geta = torch_log(T_tensor + 1e-14) + eta_d
#     Gtheta = nnf_softmax(geta, dim=-1)[,,,,,,1:-2]$unsqueeze(-1)
    
#     tmp = Gtheta$matmul(Gtheta$transpose(7,8)) - torch_diag_embed(Gtheta$squeeze())
#     tmp = tmp * Y_d$view(c(3,3,16,4,2,96,1,1))
#     # tmp$sum(c(1,2,3,4,5,6))
#     hess = Sigma_inv - Y_d$sum()*(theta_d$matmul(theta_d$transpose(1,2)) - torch_diag_embed(theta_d$squeeze()))
#     nu = hess$inverse()

#   return(nu)
# }


## Calculate our own gradient descent

update_eta_Delta <- function(T0, covs, eta, Sigma, Y,Xi, X, hyp){
  SigmaInv = Sigma$inverse()
  mu = Xi$matmul(X$transpose(1,2))
  lr = hyp$lr
  max_iter = hyp$max_iter
  tol = hyp$tol

  D = ncol(eta)
  Delta = torch_empty(c(D,K-1, K-1), device=device)
  batches = msplit(1:D, ceiling(D/50))
  for (b in batches){
      yphi_ = yphi(covs=covs, T0 = T0, Y= Y[..,b,drop=FALSE], 
                  missing_rate = make_m__(Y[..,b,drop=FALSE]), X = X[b,,drop=FALSE], context=TRUE,eta = eta[,b,drop=FALSE])
      lp = Laplace_fit(eta[,b,drop=FALSE], mu[,b,drop=FALSE],yphi_,SigmaInv,max_iter, tol, lr)
      Delta[b] = lp$Delta
      eta[,b] = lp$eta
      gc()
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
    if(stop_theta(old_grad,g$mean()$item(),tol)){
      break
    } else {
      old_grad = g$mean()$item()
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
      eta_d = torch_cat(c(eta[,i], torch_zeros(1,device=device)), dim=1)
      theta = nnf_softmax(eta_d, dim=1)[1:-2]$reshape(c(-1,1))
      hess = SigmaInv - Yn[i,]*(theta$matmul(theta$transpose(1,2)) - torch_diag_embed(theta$squeeze()))
      nu[i] = hess$inverse()
  }
  return(nu)
}

adam_optim0 <- function(eta,s,r, lr,grad, it, rho1=0.2, rho2=0.999, delta = 1e-10){
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