## Functions to update T0 and F tensors


tnf <- torch::nn_module(
    classname = "tnf",
    initialize = function(yphi_tensor, T0, factors, m_, factor_dim = c(2,2,16,4,2), coordinate_ascent = FALSE){
        self$yphi = yphi_tensor
        self$m_ = m_
        self$D = yphi_tensor$size()[6]
        self$K = yphi_tensor$size()[8]
        self$V = yphi_tensor$size()[7]
        self$factor_dim = factor_dim
        self$nepi = factor_dim[3]
        self$ca = coordinate_ascent

        if (!(self$ca)){
            self$cl_ = nn_parameter(logit_op(T0[1,1]))
            self$cg_ = nn_parameter(logit_op(T0[1,2]))
            self$tl_ = nn_parameter(logit_op(T0[2,1]))
            self$tg_ = nn_parameter(logit_op(T0[2,2]))
        }

        self$t_ = nn_parameter(logit_op(factors$bt))
        self$r_ = nn_parameter(logit_op(factors$br))

        #   self$e_ = nn_parameter(logit_op(factors$epi))
        #   self$n_ = nn_parameter(logit_op(factors$nuc))
        #   self$c_ = nn_parameter(logit_op(factors$clu))

        self$e_ = logit_op(yphi_tensor$sum(dim=c(1,2,4,5,-3,-2)))
        self$n_ = logit_op(yphi_tensor$sum(dim=c(1,2,3,5,-3,-2)))
        self$c_ = logit_op(yphi_tensor$sum(dim=c(1,2,3,4,-3,-2)))


    },
    forward = function(){
        if (!(self$ca)){
            self$cl0_ = torch_cat(c(self$cl_, torch_zeros(1, self$K, device=device)), dim=1)
            self$cl = nnf_softmax(self$cl0_, dim=1)

            self$cg0_ = torch_cat(c(self$cg_, torch_zeros(1, self$K, device=device)), dim=1)
            self$cg = nnf_softmax(self$cg0_, dim=1)

            self$tl0_ = torch_cat(c(self$tl_, torch_zeros(1, self$K, device=device)), dim=1)
            self$tl = nnf_softmax(self$tl0_, dim=1)

            self$tg0_ = torch_cat(c(self$tg_, torch_zeros(1, self$K, device=device)), dim=1)
            self$tg = nnf_softmax(self$tg0_, dim=1)
        } else {
            ## To accomodate R scoping, the update_t function can be implemented here
            t0_ = torch_cat(c(self$t_, torch_zeros(1,self$K, device=device)), dim=1)
            t = nnf_softmax(t0_, dim=1)$detach()[1]
            
            r0_ = torch_cat(c(self$r_, torch_zeros(1,self$K, device=device)), dim=1)
            r = nnf_softmax(r0_,dim=1)$detach()[1]

            yphi_ = self$yphi$sum(dim=c(3,4,5,6))

            self$cl = yphi_[1,1] + t*yphi_[3,1] + r*yphi_[1,3] + t*r*yphi_[3,3]
            self$cl = self$cl/self$cl$sum(dim=1, keepdim=TRUE)

            self$cg = yphi_[1,2] + t*yphi_[3,2] + (1-r)*yphi_[1,3] + t*(1-r)*yphi_[3,3]
            self$cg = self$cg/self$cg$sum(dim=1, keepdim=TRUE)

            self$tl = yphi_[2,1] + (1-t)*yphi_[3,1] + r*yphi_[1,3] + (1-t)*r*yphi_[3,3]
            self$tl = self$tl/self$tl$sum(dim=1, keepdim=TRUE)

            self$tg = yphi_[2,2] + (1-t)*yphi_[3,2] + (1-r)*yphi_[2,3] + (1-t)*(1-r)*yphi_[3,3]
            self$tg = self$tg/self$tg$sum(dim=1, keepdim=TRUE)
        }

        t0_ = torch_cat(c(self$t_, torch_zeros(1,self$K, device=device)))
        self$t = nnf_softmax(t0_, dim=1)
        r0_ = torch_cat(c(self$r_, torch_zeros(1,self$K, device=device)))
        self$r = nnf_softmax(r0_, dim=1)

        e0_ = torch_cat(c(self$e_, torch_zeros(1, self$K, device=device)), dim=1)
        self$e = nnf_softmax(e0_, dim=1)

        n0_ = torch_cat(c(self$n_, torch_zeros(1, self$K, device=device)), dim=1)
        self$n = nnf_softmax(n0_, dim=1)

        c0_ = torch_cat(c(self$c_, torch_zeros(1, self$K, device=device)), dim=1)
        self$c = nnf_softmax(c0_, dim=1)

        T0 = torch_stack(c(self$cl, self$cg, self$tl, self$tg))$reshape(c(2,2,-1, self$K))

        T_tensor <- stack(T0, bt= self$t, br=self$r)
        factors_ = list(bt = self$t, br = self$r, epi = self$e, nuc=self$n, clu=self$c)
        F_tensor <- factors_to_F(factors_, factor_dim = self$factor_dim, missing_rate = self$m_)

        pred = T_tensor$matmul(torch_diag_embed(F_tensor))
        loss_val =  -(self$yphi$sum(dim=-3)*torch_log(pred + 1e-14))$sum()/(self$D*self$K)
        weight =  1
        reg = covariance_regularizer(t0_, r0_,e0_, n0_, c0_, self$factor_dim)
        return( loss_val + weight*reg)
    }
)

covariance_regularizer <- function(t0_, r0_,e0_, n0_, c0_,factor_dim){
    t = nnf_softmax(t0_, dim=1)
    r = nnf_softmax(r0_,dim=1)
    e = nnf_softmax(e0_, dim=1)
    n = nnf_softmax(n0_, dim=1)
    c = nnf_softmax(c0_, dim=1)

    Ct = torch_mm(t$transpose(1,2), t) / 2
    Cr = torch_mm(r$transpose(1,2), r) / 2
    Ce = torch_mm(e$transpose(1,2), e) / 2
    Cn = torch_mm(n$transpose(1,2), n) / 2
    Cc = torch_mm(c$transpose(1,2), c) / 2

    covariance_penalty = torch_square( Ct - torch_diag(torch_diag(Ct)))$sum()/2 + 
                    torch_square(Cr - torch_diag(torch_diag(Cr)))$sum()/2 +
                    torch_square(Ce - torch_diag(torch_diag(Ce)))$sum()/factor_dim[3] +
                    torch_square(Cn - torch_diag(torch_diag(Cn)))$sum()/factor_dim[4] +
                    torch_square(Cc - torch_diag(torch_diag(Cc)))$sum()/factor_dim[5]

    return(covariance_penalty)

}

update_t <- function(factors, T0, yphi_tensor){
    K = yphi_tensor$size()[8]
    t_ = logit_op(factors$bt)
    t0_ = torch_cat(c(t_, torch_zeros(1,K, device=device)), dim=1)
    t = nnf_softmax(t0_, dim=1)$detach()[1]
    
    r_ = logit_op(factors$br)
    r0_ = torch_cat(c(r_, torch_zeros(1,K, device=device)), dim=1)
    r = nnf_softmax(r0_,dim=1)$detach()[1]

    yphi_ = yphi_tensor$sum(dim=c(3,4,5,6))

    cl = yphi_[1,1] + t*yphi_[3,1] + r*yphi_[1,3] + t*r*yphi_[3,3]
    cl = cl/cl$sum(dim=1, keepdim=TRUE)

    cg = yphi_[1,2] + t*yphi_[3,2] + (1-r)*yphi_[1,3] + t*(1-r)*yphi_[3,3]
    cg = cg/cg$sum(dim=1, keepdim=TRUE)

    tl = yphi_[2,1] + (1-t)*yphi_[3,1] + r*yphi_[1,3] + (1-t)*r*yphi_[3,3]
    tl = tl/tl$sum(dim=1, keepdim=TRUE)

    tg = yphi_[2,2] + (1-t)*yphi_[3,2] + (1-r)*yphi_[2,3] + (1-t)*(1-r)*yphi_[3,3]
    tg = tg/tg$sum(dim=1, keepdim=TRUE)

    T0 = torch_stack(c(cl, cg, tl, tg))$reshape(c(2,2, -1, K))

    return(T0)
}

update_f <- function(tmp_mod,max_iter, lr, tol){
    ## Input:
    ##      tmp_mod is a torch::nn_module to be optimized of class tnf
    old_loss_ = 1e10
    inc_loss_ = 0
    optimizer = optim_adam(tmp_mod$parameters, lr=lr)
    convergence = FALSE
    for (i in 1:max_iter){
        while(convergence ==FALSE){
            optimizer$zero_grad()
            loss = tmp_mod()
            loss$backward()
            optimizer$step()
            convergence_res = stop_crit(old_loss = old_loss_, 
                inc_loss = inc_loss_, new_loss = loss$item(), tol = tol, end="global")
            old_loss_ = convergence_res$loss 
            inc_loss_ = convergence_res$inc_loss
            convergence = convergence_res$convergence
        }
    }
    return(list(bt  = tmp_mod$t$detach(), br = tmp_mod$r$detach(),
        epi = tmp_mod$e$detach(), nuc = tmp_mod$n$detach(), clu = tmp_mod$c$detach()))

}

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


tnf_fit <- function(factors, T0, yphi_tensor, m_, coordinate_ascent = FALSE){
    tmp_mod = tnf(yphi_tensor, T0, factors, m_, factor_dim = c(2,2,16,4,2), coordinate_ascent = coordinate_ascent)
    lr = 3e-2
    max_iter = 1000
    tol = list(abs=1e-2, ratio = 1e-3)
    old_loss_ = -1e10
    inc_loss_ = 0
    convergence = FALSE
    it = 0
    if ( coordinate_ascent){
        while (convergence == FALSE && it <= max_iter){
            it = it +1
            # T0_ = update_t(factors, T0, yphi_tensor)
            factors = update_f(tmp_mod, 100, lr, tol)
            new_loss = tmp_mod()
            convergence_res = stop_crit(old_loss = old_loss_, 
            inc_loss = inc_loss_, new_loss = new_loss$item(), tol = tol, end="global")
            old_loss_ = convergence_res$loss 
            inc_loss_ = convergence_res$inc_loss
            convergence = convergence_res$convergence
        }
    } else {
        optimizer = optim_adam(tmp_mod$parameters, lr = lr)
        old_loss_ = 1e10
        while(convergence == FALSE && it <= 1:max_iter){
            if(it == max_iter){
                message("Improve max_iter tnf")
            }
            optimizer$zero_grad()
            new_loss = tmp_mod()
            new_loss$backward()
            optimizer$step()
            convergence_res = stop_crit(old_loss = old_loss_, 
            inc_loss = inc_loss_, new_loss = new_loss$item(), tol = tol, end="global")
            old_loss_ = convergence_res$loss 
            inc_loss_ = convergence_res$inc_loss
            convergence = convergence_res$convergence
        }
    }

    factors = list(bt  = tmp_mod$t$detach(), br = tmp_mod$r$detach(),
        epi = tmp_mod$e$detach(), nuc = tmp_mod$n$detach(), clu = tmp_mod$c$detach())
    
    cl = tmp_mod$cl$detach()
    cg = tmp_mod$cg$detach()
    tl = tmp_mod$tl$detach()
    tg = tmp_mod$tg$detach()

    return(list(factors=factors, cl = cl, cg= cg, tl = tl, tg=tg))
}

update_TnF <- function(eta, factors, T0, X, Y, context = TRUE, missing_rate = NULL, weight, coordinate_ascent = FALSE){

    yphi_tensor = yphi(eta, factors, T0, X, Y, context, missing_rate)

    res_tnf_fit = tnf_fit(factors, T0, yphi_tensor, missing_rate, coordinate_ascent = coordinate_ascent)
    
    T0[1,1] = weight*res_tnf_fit$cl + (1-weight)*T0[1,1]
    T0[1,2] = weight*res_tnf_fit$cg + (1-weight)*T0[1,2]
    T0[2,1] = weight*res_tnf_fit$tl + (1-weight)*T0[2,1]
    T0[2,2] = weight*res_tnf_fit$tg + (1-weight)*T0[2,2]

    for (k in names(factors)){
        factors[[k]] = weight*res_tnf_fit$factors[[k]] + (1-weight)*factors[[k]]
    }

    return(list(T0= T0, factors = factors))


}