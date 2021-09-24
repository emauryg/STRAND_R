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


## Set up dataset for dataloader

y_phi_dataset <- torch::dataset(
    name = "y_phi_dataset",
    initialize = function(Y, lambda, T_tensor, F_tensor){
        self$Y = Y$transpose(-1,-2)$unsqueeze(-1)

        ## compute phi
        self$phi = Phi(lambda, T_tensor, F_tensor)
        self$yphi = (self$Y*self$phi)$sum(dim=-3)
    },
    .getitem = function(i){
        if(length(i) >1){
                return(list(Y = self$Y[..,i,,],
                            yphi=(self$Y[..,i,,]*self$phi[..,i,,])$sum(dim=-3),
                    ))
        } else {
            return(list(Y = self$Y[..,i,,],
                    yphi=(self$Y[..,i,,]*self$phi[..,i,,])))
        }
        
    },
    .length = function(){
        self$Y$size(dim=-3)
    }
)


## module to compute our tensor reconstruction prediction
tnf_fit <- function(factors, T0,Y, tau,eta){
    
    tnf <- torch::nn_module(
    classname = "tnf",
    initialize = function(enc_start, T0, factors){
        
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
    forward = function(Y){
        m_ = make_m__(Y)
        K = ncol(self$e_)
        D = Y$size(dim=-3)
        factor_dim = c(2,2,16,4,2)
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
        return(pred)
    },
    # loss = function(input, target){
    #     pred0 <- ctx$model(input)
    #     gc()
    #     loss = -(target*torch_log(pred0 + 1e-20))$sum()/(D*K)
    # }
    step = function(){
        pred0 = ctx$model(ctx$input)
        opt = ctx$optimizer
        loss <- -(target*torch_log(pred0 + 1e-20))$sum()/(D*K)
        if(ctx$training){
            opt$zero_grad()
            loss$backward()
            opt$step()
            gc()
        }
    }

    )

    enc_start_func <- function(Y,phi){
        D = Y$size(dim=-3)
        yphi_sum_e = 0
        yphi_sum_n = 0
        yphi_sum_c = 0
        for (d in 1:D){
            yphi_sum_e = yphi_sum_e + (Y[..,d,,]*phi[..,d,,])$sum(dim=c(1,2,4,5,-2))
            yphi_sum_n = yphi_sum_n + (Y[..,d,,]*phi[..,d,,])$sum(dim=c(1,2,3,5,-2))
            yphi_sum_c = yphi_sum_c + (Y[..,d,,]*phi[..,d,,])$sum(dim=c(1,2,3,4,-2))
        }
        return(list(e = yphi_sum_e, n = yphi_sum_n, c = yphi_sum_c))
    }

    ## Compute training
    D = Y$size(dim=-1)
    train_indices <- sample(1:D, ceiling(0.8*D))
    valid_indices <- setdiff(1:D, train_indices)

    T_tensor = stack(T0 = T0, bt = factors$bt, br = factors$br)
    F_tensor = factors_to_F(factors=factors, missing_rate = make_m__(Y))

    train_ds <- y_phi_dataset(Y[..,train_indices,drop=FALSE], 
                            eta[,train_indices,drop=FALSE],
                            T_tensor, F_tensor)
    valid_ds <- y_phi_dataset(Y[..,valid_indices,drop=FALSE], 
                            eta[,valid_indices,drop=FALSE],
                            T_tensor, F_tensor)

    train_dl <- train_ds %>% dataloader(batch_size = 64, shuffle = TRUE)

    valid_dl <- valid_ds %>% dataloader(batch_size = 64, shuffle = FALSE)

    enc_start = enc_start_func(train_ds$Y, train_ds$phi)

    # print_callback <- luz::luz_callback(
    #     name = "print_callback",
    #     initialize = function(message){
    #         self$message <- message
    #     },
    #     on_train_batch_end = function(){
    #         cat("Iteration ", ctx$iter,"\n")
    #     },
    #     on_epoch_end = function(){
    #         cat(self$message, "\n")
    #     }
    # )

    early_callback <- luz::luz_callback_early_stopping(
        monitor = "valid_loss",
        min_delta = 1e-2,
        patience = 2,
        mode= "zero",
        baseline=1e10)

    fitted <- tnf %>% luz::setup(
        optimizer = optim_adam) %>%
        luz::set_hparams(enc_start, T0, factors) %>%
        luz::set_opt_hparams(lr = 0.05) %>%
        luz::fit(train_dl, epochs = 10000, valid_data = valid_dl,
            callbacks = list(early_callback), verbose = FALSE)

    cl = fitted$model$cl$detach()
    cg = fitted$model$cg$detach()
    tl = fitted$model$tl$detach()
    tg = fitted$model$tg$detach()

    factors = list(bt = fitted$model$t$detach(),
                    br = fitted$model$r$detach(),
                    epi = fitted$model$e$detach(),
                    nuc = fitted$model$n$detach(),
                    clu = fitted$model$c$detach())

    gc()
    return(list(factors=factors, cl=cl, cg=cg, tl=tl, tg=tg))
}