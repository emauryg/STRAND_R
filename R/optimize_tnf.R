## Functions to update T0 and F tensors


tnf <- torch::nn_module(
    classname = "tnf",
    initialize = function(yphi_tensor, T0, factors){
        # self$yphi = yphi_tensor
        # self$m_ = m_
        # self$D = yphi_tensor$size()[6]
        # self$K = yphi_tensor$size()[8]
        # self$V = yphi_tensor$size()[7]
        # self$factor_dim = factor_dim
        self$cl_ = nn_parameter(logit_op(T0[1,1]))
        self$cg_ = nn_parameter(logit_op(T0[1,2]))
        self$tl_ = nn_parameter(logit_op(T0[2,1]))
        self$tg_ = nn_parameter(logit_op(T0[2,2]))

        self$t_ = nn_parameter(logit_op(factors$bt))
        self$r_ = nn_parameter(logit_op(factors$br))

        self$e_ = logit_op(yphi_tensor$sum(dim=c(1,2,4,5,-3,-2)))
        self$n_ = logit_op(yphi_tensor$sum(dim=c(1,2,3,5,-3,-2)))
        self$c_ = logit_op(yphi_tensor$sum(dim=c(1,2,3,4,-3,-2)))
    },
    forward = function(m_, factor_dim = c(2,2,16,4,2), yphi_tensor,tau=0.01){
        D =  yphi_tensor$size()[6]
        K =  yphi_tensor$size()[8]
        V = yphi_tensor$size()[7]

        self$cl0_ = torch_cat(c(self$cl_, torch_zeros(1, K, device=device)), dim=1)
        self$cl = nnf_softmax(self$cl0_, dim=1)
        self$cg0_ = torch_cat(c(self$cg_, torch_zeros(1, K, device=device)), dim=1)
        self$cg = nnf_softmax(self$cg0_, dim=1)
        self$tl0_ = torch_cat(c(self$tl_, torch_zeros(1, K, device=device)), dim=1)
        self$tl = nnf_softmax(self$tl0_, dim=1)
        self$tg0_ = torch_cat(c(self$tg_, torch_zeros(1, K, device=device)), dim=1)
        self$tg = nnf_softmax(self$tg0_, dim=1)

        t0_ = torch_cat(c(self$t_, torch_zeros(1,K, device=device)))
        self$t = nnf_softmax(t0_, dim=1)

        r0_ = torch_cat(c(self$r_, torch_zeros(1,K, device=device)))
        self$r = nnf_softmax(r0_, dim=1)

        e0_ = torch_cat(c(self$e_, torch_zeros(1, K, device=device)), dim=1)
        self$e = nnf_softmax(e0_, dim=1)

        n0_ = torch_cat(c(self$n_, torch_zeros(1, K, device=device)), dim=1)
        self$n = nnf_softmax(n0_, dim=1)

        c0_ = torch_cat(c(self$c_, torch_zeros(1, K, device=device)), dim=1)
        self$c = nnf_softmax(c0_, dim=1)

        T0 = torch_stack(c(self$cl, self$cg, self$tl, self$tg))$reshape(c(2,2,-1, K))

        T_tensor <- stack(T0, bt= self$t, br=self$r)
        factors_ = list(bt = self$t, br = self$r, epi = self$e, nuc=self$n, clu=self$c)
        F_tensor <- factors_to_F(factors_, factor_dim = factor_dim, missing_rate = m_)
        pred = T_tensor$matmul(torch_diag_embed(F_tensor))
        loss_val =  -(yphi_tensor$sum(dim=-3)*torch_log(pred + 1e-14))$sum()/(D*K)
        weight =  tau
        Cr = torch_mm(self$r$transpose(1,2), self$r) / 2
        Ct = torch_mm(self$t$transpose(1,2), self$t) / 2        
        Ce = torch_mm(self$e$transpose(1,2), self$e) / factor_dim[3]
        Cn = torch_mm(self$n$transpose(1,2), self$n) / factor_dim[4]
        Cc = torch_mm(self$c$transpose(1,2), self$c) / factor_dim[5]
        # mu_r = self$r$mean(dim=1)
        # mu_t = self$t$mean(dim=1)
        # mu_e = self$e$mean(dim=1)
        # mu_n = self$n$mean(dim=1)
        # mu_c = self$c$mean(dim=1)
        # Cr = torch_mm( (self$r - mu_r)$transpose(1,2), self$r - mu_r) / 2
        # Ct = torch_mm( (self$t - mu_t)$transpose(1,2), self$t - mu_t) / 2        
        # Ce = torch_mm( (self$e - mu_e)$transpose(1,2), self$e - mu_e) / factor_dim[3]
        # Cn = torch_mm( (self$n - mu_n)$transpose(1,2), self$n - mu_n) / factor_dim[4]
        # Cc = torch_mm( (self$c - mu_c)$transpose(1,2), self$c - mu_c) / factor_dim[5]

        reg = torch_square( Ct - torch_diag(torch_diag(Ct)))$sum()/2 + 
                torch_square(Cr - torch_diag(torch_diag(Cr)))$sum()/2 +
                torch_square(Ce - torch_diag(torch_diag(Ce)))$sum()/factor_dim[3] +
                torch_square(Cn - torch_diag(torch_diag(Cn)))$sum()/factor_dim[4] +
                torch_square(Cc - torch_diag(torch_diag(Cc)))$sum()/factor_dim[5]
        # reg = ( Ct - torch_diag(torch_diag(Ct)))$sum()/2 + 
        #     (Cr - torch_diag(torch_diag(Cr)))$sum()/2 +
        #     (Ce - torch_diag(torch_diag(Ce)))$sum()/factor_dim[3] +
        #     (Cn - torch_diag(torch_diag(Cn)))$sum()/factor_dim[4] +
        #     (Cc - torch_diag(torch_diag(Cc)))$sum()/factor_dim[5]

        return( loss_val + weight*reg)
    }
)





stop_crit <- function(old_loss, inc_loss, new_loss, tol, patience = 5, end = NULL){
    abs_cri = FALSE
    rat_cri = FALSE

    if (end == "global"){
        if (new_loss > old_loss){
            inc_loss = inc_loss + 1
            if(inc_loss >= patience){
                return(list(convergence = TRUE, loss=new_loss, inc_loss = inc_loss) )
            }
        }
        if (abs(new_loss - old_loss) < tol$abs){
            abs_cri = TRUE
        }
        if (abs(new_loss - old_loss)/(abs(old_loss) + 1e-20) < tol$ratio ){
            rat_cri = TRUE
        }
    } else {
        if ( abs( new_loss - old_loss)/(abs(old_loss)+ 1e-20) < tol$ratio){
            rat_cri = TRUE
        }
    }
    if (abs_cri & rat_cri){
        return(list(convergence=TRUE, loss= new_loss, inc_loss = inc_loss))
    } else {
        if (end == 'global'){
            old_loss = new_loss
        } else {
            old_loss = new_loss ## this might be removed since scoping in R is not the same as python
        }
        return(list(convergence=FALSE, loss= new_loss, inc_loss = inc_loss))
    }
}


tnf_fit <- function(factors, T0, yphi_tensor, m_,tau=0.01){
    tmp_mod = tnf(yphi_tensor, T0, factors)
    lr = 5e-2
    max_iter = 1000
    min_iter = 100
    tol = list(abs=1e-2, ratio = 1e-3)
    old_loss_ = -1e10
    inc_loss_ = 0
    convergence = FALSE
    it = 0
    optimizer = optim_adam(tmp_mod$parameters, lr = lr)
    old_loss_ = 1e10
    while(convergence == FALSE & it <= max_iter){
        it = it + 1
        if(it == max_iter){
            message("Improve max_iter tnf")
        }
        optimizer$zero_grad()
        new_loss = tmp_mod(m_, factor_dim = c(2,2,16,4,2), yphi_tensor,tau)
        new_loss$backward()
        optimizer$step()
        if(it >= min_iter){
            convergence_res = stop_crit(old_loss = old_loss_, 
            inc_loss = inc_loss_, new_loss = new_loss$item(), tol = tol, end="global")
            old_loss_ = convergence_res$loss 
            inc_loss_ = convergence_res$inc_loss
            convergence = convergence_res$convergence
        }

        gc()
  
    }
    
    factors = list(bt  = tmp_mod$t$detach(), br = tmp_mod$r$detach(),
        epi = tmp_mod$e$detach(), nuc = tmp_mod$n$detach(), clu = tmp_mod$c$detach())
    
    cl = tmp_mod$cl$detach()
    cg = tmp_mod$cg$detach()
    tl = tmp_mod$tl$detach()
    tg = tmp_mod$tg$detach()

    return(list(factors=factors, cl = cl, cg= cg, tl = tl, tg=tg))
}

update_TnF <- function(eta, factors, T0, X, Y, context = FALSE, missing_rate = NULL, weight, tau=0.01){

    yphi_tensor = yphi(eta, factors, T0, X, Y, context, missing_rate)

    res_tnf_fit = tnf_fit(factors, T0, yphi_tensor, missing_rate, tau)


    T0[1,1] = res_tnf_fit$cl 
    T0[1,2] = res_tnf_fit$cg 
    T0[2,1] = res_tnf_fit$tl 
    T0[2,2] = res_tnf_fit$tg 

    for (k in names(factors)){
        factors[[k]] = res_tnf_fit$factors[[k]] 
    }

    return(list(T0= T0, factors = factors))


}