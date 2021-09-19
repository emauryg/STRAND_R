## helper functions

logit_op <- function(tens, eps=1e-20){
    ## Performs logit transformation
    ## Input: 
    ##      tens ("torch tensor"): The input tensor
    ##      eps (float): The small value for numerical stability
    ## Output:
    ##      logit(tensor)

    denom = tens[-1]
    denom[denom < eps] = eps
    odd_ratio = tens[1:-2]/denom
    odd_ratio[odd_ratio < eps] = eps
    return(torch_log(odd_ratio))
}


Phi <- function(lam, T_tensor, F_tensor, sam_covs = TRUE, eps=1e-20){
    ## Computes Phi tensor
    ## Input:
    ##      T_tensor (torch_tensor): stacked T tensor with dimensions 3x3x16x4x2
    ##      F_tensor (torch_tensor): a 3x3x16x4x2x96 tensor
    ##      lam (torch_tensor): optimized eta tensor, which is KxD, D is the number of samples
    ## Output:
    ##      phi, a torch_tensor of dimension of 3x3x16x4x2x100x96x5

    D = ncol(lam)
    if (sam_covs){
        lam = torch_cat(c(lam, torch_zeros(1, D, device=device)), dim=1)
    } else {
        lam = torch_log(lam + eps)
    }

    lam = lam$transpose(1,2)
    
    phi = torch_log(T_tensor)$unsqueeze(-3) + lam$unsqueeze(-2) + torch_log(F_tensor)$unsqueeze(-2)$unsqueeze(-2)
    rm(lam)
    gc()
    return(nnf_softmax(phi, dim=-1))
}

YPhi <- function(Y, lam, T_tensor, F_tensor, sam_covs = TRUE, context = FALSE){
    ## Computes the product of Y and phi

    phi = Phi(lam, T_tensor, F_tensor, sam_covs)

    Y = Y$transpose(-1,-2)
    if (context){ 
        return((Y$unsqueeze(-1)*phi)$sum(dim=c(1,2,3,4,5,-2)))
        
    } else{
        return( (Y$unsqueeze(-1)*phi))
    }
}

yphi <- function(eta, covs, T0, X, Y, context=FALSE, missing_rate=NULL){
    T_tensor = stack(T0 = T0, bt = covs$bt, br = covs$br)
    F_tensor = factors_to_F(factors=covs, missing_rate = missing_rate)

    if ( !is.null(X)) {
        yphi = YPhi(Y, eta, T_tensor, F_tensor, sam_covs = TRUE, context = context)
    } else {
        yphi = YPhi(Y, eta, T_tensor, F_tensor, sam_covs = FALSE, context = context)
    }

    return(yphi)

}

#' Generate stacked T0 tensor
#' 
#' @param T0 , tensor 
#' @param bt , transcriptional bias tensor
#' @param br , replication bias tensor
#' @export
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

  T_tensor[1,1] = CL ; T_tensor[2,1] = TL ; T_tensor[3,1] = T0_L
  T_tensor[1,2] = CG ; T_tensor[2,2] = TG ; T_tensor[3,2] = T0_G
  T_tensor[1,3] = T0_C_ ; T_tensor[2,3] = T0_T_ ; T_tensor[3,3] = T0__

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

#' Function to compute the product of the stacked tensor T0, and F cofactors
#'
#' @param T0 a tensor
#' @param F a list of covariate factors
#' @return T tensor 
#' @export
tf <- function(T0, covs, missing_rate=NULL){
    ## Output:
    ##      TF torch_tensor with dimension 3x3x16x4x2x96xK

    T_tensor = stack(T0 = T0, bt = covs$bt, br = covs$br)
    F_tensor = factors_to_F(factors=covs, missing_rate = missing_rate)

    return(T_tensor$matmul(torch_diag_embed(F_tensor)))

}

make_m_ <- function(Y){
    ## Computes the missing rate
    ## m_ 2x2xD tensor
    y_tr = Y$sum(dim=c(3,4,5,6,7))
    m00 = y_tr[1:3,1:3]$sum(dim=c(1,2))/y_tr$sum(dim=c(1,2))
    m01 = y_tr[1:3,3]$sum(dim=c(1))/y_tr$sum(dim= c(1,2))
    m10 = y_tr[3,1:3]$sum(dim=c(1))/y_tr$sum(dim=c(1,2))
    m11 = y_tr[3,3]/y_tr$sum(dim=c(1,2))

    m_ = torch_stack(c(m00, m01, m10, m11))$reshape(c(2,2,-1))
    return(m_)
}

#' Calculate the missing rates
#'
#' @param Y count tensor
#' @export 
make_m__ <- function(Y){
    ## Computes the missing rate
    ## m__ 2x2 tensor
    y_tr = Y$sum(dim=c(3,4,5,6,7))
    m00 = y_tr[1:3,1:3]$sum(dim=c(1,2))/y_tr$sum()
    m01 = y_tr[1:3,3]$sum(dim=c(1))/y_tr$sum()
    m10 = y_tr[3,1:3]$sum(dim=c(1))/y_tr$sum()
    m11 = y_tr[3,3]/y_tr$sum()

    m_ = torch_stack(c(m00, m01, m10, m11))$reshape(c(2,2,-1))$squeeze()
    return(m_)
}

#' Compute the estimated count matrix from tenosr signature model
#'
#' @param mod_ts fitted model from ts_fit
#' @export
calculate_Chat_ts <- function(mod_ts){
    E = torch_exp(mod_ts$VIparam$E$clone())
    V = anno_dims$V
    K = length(mod_ts$Bparam$factors$bt)

    ## Calculate T1
    cl = mod_ts$Bparam$T0[1,1]
    cg = mod_ts$Bparam$T0[1,2]
    tl = mod_ts$Bparam$T0[2,1]
    tg = mod_ts$Bparam$T0[2,2]

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
    bt_ = mod_ts$Bparam$factors$bt$clone()
    br_ = mod_ts$Bparam$factors$br$clone()
    at_ = mod_ts$Bparam$factors$at$clone()
    ar_ = mod_ts$Bparam$factors$ar$clone()
    B = torch_stack(c(bt_ + br_, bt_- br_, bt_,
                    -bt_ + br_, -bt_ - br_, -bt_,
                    br_, -br_, torch_zeros(K, device=device)))$reshape(c(3,3,1,K))
    B = torch_exp(B)
      
      

    ## Calculate A
    A = torch_stack(c(at_ + ar_, at_ - ar_, at_,
                  at_ + ar_, -at_ - ar_, at_, 
                  ar_, ar_, torch_zeros(K, device=device)))$reshape(c(3,3,1,K))
    A = torch_exp(A)

    ## Calculate K_epi, K_nuc, and K_clu
    K_epi = mod_ts$Bparam$factors$epi$clone()
    K_nuc= mod_ts$Bparam$factors$nuc$clone()
    K_clu = mod_ts$Bparam$factors$clu$clone()

    ## Calculate T_strand
    T_strand = T1 * B * A

    ## Calculate K_tensor
    K_tensor = K_epi$view(c(1,1,-1,1,1,K)) * K_nuc$view(c(1,1,1,-1,1,K)) *
      K_clu$view(c(1,1,1,1,-1, K))

    ## Calculate T_tensor
    T_tensor = T_strand$view(c(3,3,1,1,1,-1,K))*K_tensor$unsqueeze(-2)
    
    ## Calculate Chat
    Chat = T_tensor$matmul(E)
    return(Chat)
}

#' Compute the deviance of a model
#'
#' @param Y is a count tensor
#' @param res is the output of running function{run_EM}. 
#' @param X is the matrix of covariates.
#' @param method which method was used {STRAND, or TENSIG}, (default: "STRAND")
#' @return deviance
#' @export
## Computing deviance
compute_deviance <- function(Y, res, return_value="deviance", X = NULL, method="STRAND"){
    ## Input:
    ##      Y = count tensor
    ##      res = list results of a STRAND run
    ##      method = method that was used to perform inference {STRAND, TensorSignature} ## TODO tensor sig approach

    tf_tensor = tf(res$Bparam$T0,res$Bparam$factors,missing_rate = make_m__(Y))
    if (method == "STRAND"){
        lam = res$VIparam$lambda
        lam = torch_cat(c(lam, torch_zeros(1,ncol(lam), device=device)), dim=1)
        theta = nnf_softmax(lam, dim=1)
        Chat = tf_tensor$matmul(theta*Y$sum(dim=c(1,2,3,4,5,6)))
    } else if(method == "TENSIG"){
        Chat = calculate_Chat_ts(res)
    }

    pred = Chat

    dev = 2*(Y*torch_log(Y/(pred+1e-10)+1e-10) -Y+pred)$sum()

    return(dev$item())
}


#' Compute the deviance curve to estimate the number of clusters
#'
#' @param num_sigs : vector of the number of signatures to use
#' @param Y : count tensor
#' @param X : matrix of covariates
#' @return data frame with the number of clusters and the deviance
#' @export
deviance_curve <- function(num_sigs = c(), Y, X){
    if(1 %in% num_sigs){
        return(message("K cannot be equal to 1. Provide numbers greater than 1."))
    }
    devs = rep(0, length(num_sigs))
    for(i in 1:length(num_sigs)){
        k = num_sigs[i]
        message("Fitting K = ",k, "\n")
        inits <- NMFinit(Y,X,k, max_iter=10000)     
        res = runEM(inits, Y, X=X)
        tmp = compute_deviance(Y, res, return_value="deviance", X = X)
        devs[i] = as_array(tmp$cpu())
    }

    deviance_df = data.frame(K = num_sigs, deviances = devs)
    return(deviance_df)
}


#' Compute the estimated effect of the model covariates on signature contribution
#'
#' @param mod0 : fitted model from STRAND
#' @param X: design matrix with the covariates you wish to estimate the effect of
#' @param niter : number of simulations to draw for the estimation of the effect (default = 1000)
#' @param sig_number: number of the signature you want to estimate the effect of (i.e. 1, or 2, etc.)
#' @param to_plot: boolean indicating whether you want to plot the effect (default = TRUE)
#' @return a list containing the summary of the coefficients including p-values, simulated draws, and ggplot object. 
#' @export
estimateEffect <- function(mod0, X, niter, sig_number, to_plot=TRUE){
    ## Simulate signature proportion from model
    sim_estimates =matrix(0, nr= niter, ncol = ncol(X))
    for(i in 1:niter){
        mu_tmp = as_array(mod0$VIparam$lambda$cpu())
        l_tmp = matrix(0, nr=nrow(mu_tmp), nc=ncol(mu_tmp))
        for(j in 1:ncol(mu_tmp)){
            ## ensure that sigma is PSD. 
            sigma_tmp = linalg_cholesky(torch_sqrt(mod0$VIparam$Delta[j]$clone()))
            sigma = sigma_tmp$matmul(sigma_tmp$transpose(1,2))
            l_tmp[,j] = rmvnorm(n=1,mean = mu_tmp[,j], sigma = as_array(sigma$cpu()))
        }
        theta_tmp = torch_tensor(l_tmp, device=device)
        theta_tmp = nnf_softmax(torch_cat(c(theta_tmp, torch_zeros(1,D, device=device)), dim=1), dim=1)
        theta_tmp = t(as_array(theta_tmp$cpu()))
        colnames(theta_tmp)=  paste0("signature",1:K)

        df_tmp = cbind(metadata[,"muts"],X, theta_tmp) 

        df_tmp = df_tmp %>% 
            mutate(mut_sig1 = signature1*muts, mut_sig2=signature2*muts, 
                   mut_sig3= signature3*muts)
        sig_name = paste0("mut_sig",sig_number)
        covar_names = paste(colnames(X)[-1],collapse=" + ")
        tmp = summary(lm(as.formula(paste0(sig_name, "~",covar_names)), data= df_tmp))
        sim_estimates[i,] = coef(tmp)[,"Estimate"]

    }
    colnames(sim_estimates) = colnames(X)
    
    ## compute p-value
    est <- colMeans(sim_estimates, na.rm=TRUE)
    se <- sqrt(apply(sim_estimates,2, function(x){stats::var(x,na.rm=TRUE)}))
    tval <- est/se
    rdf = nrow(X) - length(est)
    p_val <- 2*stats::pt(abs(tval), rdf, lower.tail=FALSE)
    low_ci = apply(sim_estimates, MARGIN = 2, function(x){quantile(x,0.025, na.rm=TRUE)})
    high_ci = apply(sim_estimates, MARGIN=2, function(x){quantile(x,0.975, na.rm=TRUE)})
    coef_summary = cbind(est,se, low_ci,high_ci, tval, p_val)
    colnames(coef_summary) = c("Estimate","Std. error","2.5% CI","97.5% CI", "t-stat","p-value")
    cat("Signature ",sig_number, "summary: \n")
    print(coef_summary)
    
    if(to_plot){
        p1 = sim_estimates %>% as.data.frame() %>% pivot_longer(everything(), names_to = "coefficient", values_to ="estimates") %>%
        group_by(coefficient) %>% 
        summarise(MAP = mean(estimates), low_ci = quantile(estimates, probs=0.025), high_ci=quantile(estimates, probs=0.975)) %>% 
        ungroup() %>%
        ggplot(aes(x=coefficient,y=MAP)) + 
            geom_pointrange(aes(ymin=low_ci, ymax=high_ci)) +
            theme_minimal(base_size = 16) + 
            geom_hline(yintercept = 0.0, linetype=2) + 
            labs(y="sSNV/sample", title=paste0("Signature ",sig_number)) + 
            coord_flip()
        return(list(summary= coef_summary,simulatedEstimates = sim_estimates, plot = p1))
    }
    
    return(list(summary = coef_summary, simulatedEstimates=sim_estimates, plot=p1))
    
}
