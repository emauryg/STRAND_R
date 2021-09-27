## Optimize tnf with mini_batch

update_TnF <- function(eta, factors, T0, X, Y, context = FALSE, missing_rate = NULL, weight, tau=0.01){
    gc()
    res_tnf_fit = tnf_fit(factors, T0, Y, tau,eta)

    gc()

    T0[1,1] = res_tnf_fit$cl 
    T0[1,2] = res_tnf_fit$cg 
    T0[2,1] = res_tnf_fit$tl 
    T0[2,2] = res_tnf_fit$tg 

    for (k in names(factors)){
        factors[[k]] = res_tnf_fit$factors[[k]] 
    }

    return(list(T0= T0, factors = factors))


}

tnf <- torch::nn_module(
    classname = "tnf",
    initialize = function(enc_start, T0, factors, tau){
        self$cl_ = nn_parameter(logit_op(T0[1,1]))
        self$cg_ = nn_parameter(logit_op(T0[1,2]))
        self$tl_ = nn_parameter(logit_op(T0[2,1]))
        self$tg_ = nn_parameter(logit_op(T0[2,2]))
        self$t_ = nn_parameter(logit_op(factors$bt))
        self$r_ = nn_parameter(logit_op(factors$br))

        self$e_ = logit_op(enc_start$e)
        self$n_ = logit_op(enc_start$n)
        self$c_ = logit_op(enc_start$c)
    },
    forward = function(m_,yphi){
        K = ncol(self$e_)
        factor_dim= c(2,2,16,4,2)
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
        -(yphi*torch_log(pred+1e-20))$sum()
    }

)

enc_start_func <- function(Y,phi){
    D = Y$size(dim=-3)
    #Y = Y$transpose(-1,-2)$unsqueeze(-1)
    yphi_sum_e = 0
    yphi_sum_n = 0
    yphi_sum_c = 0
    batch_size=128
    batch_idx = msplit(1:D, floor(D/batch_size))
    for (d in batch_idx){
        yphi_sum_e = yphi_sum_e + (Y[..,d,,]*phi[..,d,,])$sum(dim=c(1,2,4,5,-2))
        yphi_sum_n = yphi_sum_n + (Y[..,d,,]*phi[..,d,,])$sum(dim=c(1,2,3,5,-2))
        yphi_sum_c = yphi_sum_c + (Y[..,d,,]*phi[..,d,,])$sum(dim=c(1,2,3,4,-2))
    }
    return(list(e = yphi_sum_e, n = yphi_sum_n, c = yphi_sum_c))
}

stop_run <- function(old_loss_, loss,tol, cur_patience){
    abs_cri = FALSE
    rat_cri = FALSE
    patience = 5 # how many iterations of doing worse to wait
    if(loss > old_loss_){
        cur_patience = cur_patience + 1
        if(cur_patience > patience){
            return(list(stop=TRUE,cur_patience=cur_patience))
        }
    }

    if (abs(loss - old_loss_)< tol$abs){
        abs_cri = TRUE
    }
    if (abs(loss - old_loss_)/(abs(old_loss_)+1e-20) < tol$rat){
        rat_cri = TRUE
    }
    if(abs_cri && rat_cri){
        return(list(stop=TRUE,cur_patience=cur_patience))
    } else(
        list(stop=FALSE,cur_patience=cur_patience)
    )
}


tnf_fit <- function(factors, T0,Y, tau,eta){
    m_ = make_m__(Y)
    T_tensor = stack(T0=T0, bt = factors$bt, br = factors$br)
    F_tensor = factors_to_F(factors=factors, missing_rate = m_)
    phi = Phi(eta, T_tensor, F_tensor)

    D = Y$size(dim=-1)
    K = phi$size(dim=-1)
    Y = Y$transpose(-1, -2)$unsqueeze(-1)

    # split train and validation set
    train_index = sample(1:D, floor(D*0.8))
    valid_index = setdiff(1:D, train_index)

    Y_train = Y[,,,,,torch_tensor(as.integer(train_index)),,]
    
    phi_train = phi[,,,,,torch_tensor(as.integer(train_index)),,]
    
    yphi_valid = (Y[,,,,,torch_tensor(as.integer(valid_index)),,]*phi[,,,,,torch_tensor(as.integer(valid_index)),,])$sum(dim=-3)
    train_size = length(train_index)
    valid_size = length(valid_index)

    enc_start = enc_start_func(Y, phi)

    lr = 5e-2
    max_iter = 1000
    min_iter = 100
    tol = list(abs=1e-2, ratio=1e-3)
    old_loss_ = 1e10

    tnf_mod = tnf(enc_start, T0, factors, tau)
    optimizer = optim_adam(tnf_mod$parameters, lr=lr)
    batch_size=128
    cur_patience = 0
    burn_period = 100
    for (i in 1:max_iter){
        optimizer$zero_grad()
        idx = sample(1:train_size, batch_size)
        yphi0 = (Y_train[,,,,,torch_tensor(as.integer(idx)),,]*phi_train[,,,,,torch_tensor(as.integer(idx)),,])$sum(dim=-3)
        loss = tnf_mod(m_, yphi0)/batch_size 
        loss$backward()
        optimizer$step()
        if (i %% burn_period == 0){
            loss = tnf_mod(m_, yphi_valid)/valid_size
            converged = stop_run(old_loss_, loss$item(),tol,cur_patience)
            if(converged$stop){
                break
            }
            old_loss_ = loss$item()
            cur_patience = converged$cur_patience
        }
        gc()
    }

    factors = list(bt = tnf_mod$t$detach(), br = tnf_mod$r$detach(), 
        epi = tnf_mod$e$detach(), nuc=tnf_mod$n$detach(), clu=tnf_mod$c$detach())
    cl = tnf_mod$cl$detach()
    cg = tnf_mod$cg$detach()
    tl = tnf_mod$tl$detach()
    tg = tnf_mod$tg$detach()
    return(list(factors=factors, cl=cl, cg=cg, tl=tl, tg=tg))
}




