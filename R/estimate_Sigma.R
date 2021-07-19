## Estimate the global Sigma parameter
update_Sigma <- function(nu, zeta, Xi, eta,X, weight=0.01){
  
  Sigma = nu$mean(dim=1)
  mu = Xi$matmul(X$transpose(1,2))
  disc = (eta - mu)$transpose(1,2)$unsqueeze(-1)
  tmp = disc$matmul(disc$transpose(-1,-2))
  
  Sigma = Sigma + tmp$mean(dim=1)

  tmp = X$matmul(zeta)$matmul(X$transpose(1,2))
  tmp = torch_diagonal(tmp, dim1=2,dim2=3)$mean(dim=2)
  Sigma = Sigma + torch_diag_embed(tmp)
  return(weight*torch_diagonal(Sigma) + (1-weight)*Sigma)
}


## Estimate the gamma sigma parameter
update_gamma_sigma <- function(gamma_sigma, zeta, Xi){
  p = ncol(Xi)
  K = nrow(Xi)

  for (k in 1:K){
    gamma_sigma[k] = torch_sqrt((torch_trace(zeta[k]) + (Xi[k]^2)$sum())/p)
  }

  return(gamma_sigma)
}

