## Estimating Xi and Zeta

## These functions estimate the variational parameter of Gamma with closed form solution (CS)
# library(nnet)
# library(Matrix)
# library(tidyverse)
EstimateXi <- torch::nn_module(
    classname = "EstimateXi",
    initialize = function(Xi, SigmaInv, gamma_sigma){
        self$Xi = nn_parameter(Xi)
        self$SigmaInv = SigmaInv
        self$alpha = 1/(2*gamma_sigma^2)
    },
    forward = function(x, y){
        ## Input:
        ##      x (torch_tensor): The design matrix X (size: "p x D")
        ##      y (torch_tensor): The variational parameter lambda (size: (K-1) x D)

        D = ncol(y)
        mu = self$Xi$matmul(x)
        loss = torch_trace((mu -y)$transpose(1,2)$matmul(self$SigmaInv$matmul(mu-y)))
        regularizer = ((self$Xi^2)$sum(dim=2))$dot(self$alpha)

        return((loss + regularizer)/D)

    }
)

update_Xi <- function(Xi, Sigma, gamma_sigma, X, lam, hyp, method = "stochastic"){
    SigmaInv = Sigma$inverse()
    if(method == "sylvester"){
        ## https://github.com/ajt60gaibb/freeLYAP/blob/master/lyap.m
        ##  Solving optimization problem by solving the Sylvester equation
        ##  AX + XB + C = 0, where we are interested in solving for matrix X
        weight =  1
        K = nrow(Sigma) + 1
        p = ncol(Xi)
        # K-1 x K-1 matrix
        A  = Sigma$matmul(torch_eye(K-1, device=device)*1/(2*gamma_sigma^2))
        # p x p matrix
        B = X$transpose(1,2)$matmul(X)
        # K-1 x p matrix
        C = -lam$matmul(X)


        A = as_array(A$cpu()); B = as_array(B$cpu()); C= as_array(C$cpu());

        # compute Schur factorizations, TA will be upper triangular. TB will be upper or lower. 
        # If TB is upper triangular, then we want to backward solve, but if it is lower we want to forward solve. 
        schurA = Schur(A)
        schurB = Schur(B)
        solve_direction = "backward"
        ZA = schurA$Q; TA = schurA$T
        schurB = Schur(B)
        ZB = schurB$Q; TB = schurB$T 
        # solve_direction = "forward"
        
        # transform the right hand side
        Fmat = t(ZA) %*% C %*% ZB

        # Diagonal mask (for speed in shifted solves)
        Ysol = matrix(0, nr = K-1, nc = p)
        pdiag = diag(TA)

        if( solve_direction ==  "backward"){
            kk = seq.int(p, 1, -1)
        } else {
            kk = 1:p 
        }

        for (k in kk){
            rhs = Fmat[,k, drop=FALSE] + Ysol %*%TB[,k, drop=FALSE]
            # find the kth column of the transformed solution
            diag(TA) = pdiag  + TB[k,k]
            ## TA might be difficult to invert if singular, let's use qr.solve().  
            Ysol[,k] = qr.solve(TA) %*% (-rhs)
        }

        Xi_new = ZA %*% Ysol %*% t(ZB) 
        Xi_new = torch_tensor(Xi_new, device=device)
        Xi = weight*Xi_new + (1-weight)*Xi
        return(Xi)

    } else if( method == "stochastic"){
        old_loss = 1e10
        lr = hyp[["lr"]]
        max_iter = hyp[['max_iter']]
        tol = hyp[['tol']]
        tmp_mod = EstimateXi(Xi, SigmaInv, gamma_sigma)
        it = 0
        converged = FALSE
        optimizer = optim_adam(tmp_mod$parameters, lr = lr)
        while(converged == FALSE && it <= max_iter){
            it = it +1
            optimizer$zero_grad()
            new_loss = tmp_mod(X$transpose(1,2), lam)
            new_loss$backward()
            optimizer$step()
            if( it >= 100){
                convergence_res = stop_xi(new_loss$item(), old_loss, tol)
                old_loss = new_loss$item()
                converged = convergence_res$convergence
            }
        }
        return(tmp_mod$parameters$Xi$detach())
    } else if(method == "multinomial"){
        ## TODO: Still need to figure out the correct way to encode the response variable. 
    }
}

stop_xi <- function(loss, old_loss, tol){
    if (abs(loss - old_loss) < tol['abs']){
        return(list(loss = loss, convergence = TRUE))
    } else {
        return(list(loss = loss, convergence=FALSE))
    }
}


## Functions to optimize zeta

update_zeta <- function(zeta, Sigma, gamma_sigma, X){
    SigmaInv = Sigma$inverse()
    K = nrow(Sigma) # note that this is actually K-1
    p = ncol(X)
    #X = X$transpose(1,2)
    for (k in 1:K) {
        zeta[k] = 1/(gamma_sigma[k]^2)*torch_eye(p, device=device) + SigmaInv[k,k]*X$transpose(1,2)$matmul(X)
    }
    return(zeta$inverse())
}