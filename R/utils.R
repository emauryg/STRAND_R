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

    phi = torch_log(T_tensor)
    phi = phi$unsqueeze(-3) + lam$unsqueeze(-2)
    phi = phi + torch_log(F_tensor)$unsqueeze(-2)$unsqueeze(-2)

    return(nnf_softmax(phi, dim=-1))
}

YPhi <- function(Y, lam, T_tensor, F_tensor, sam_covs = TRUE, context = TRUE){
    ## Computes the product of Y and phi

    phi = Phi(lam, T_tensor, F_tensor, sam_covs)

    Y = Y$transpose(-1,-2)
    if (context){ 
        return(Y$unsqueeze(-1)*phi)
    } else{
        return((Y$unsqueeze(-1)*phi)$sum(dim=c(1,2,3,4,5,-2)))
    }
}

yphi <- function(eta, covs, T0, X, Y, context=TRUE, missing_rate=NULL){
    T_tensor = stack(T0 = T0, bt = covs$bt, br = covs$br)
    F_tensor = factors_to_F(factors=covs, missing_rate = missing_rate)

    if ( !is.null(X)) {
        yphi = YPhi(Y, eta, T_tensor, F_tensor, sam_covs = TRUE, context = context)
    } else {
        yphi = YPhi(Y, eta, T_tensor, F_tensor, sam_covs = FALSE, context = context)
    }

    return(yphi)

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


#' Compute the deviance of a model
#'
#' @param Y is a count tensor
#' @param res is the output of running function{run_EM}. 
#' @param X is the matrix of covariates.
#' @return deviance
compute_deviance <- function(Y, res, return_value="deviance", X = NULL){
    ## Input:
    ##      Y = count tensor
    ##      res = list results of a HDsig run
    ##      method = method that was used to perform inference {HDsig, TensorSignature} ## TODO tensor sig approach
    n_epi = nrow(res$Bparam$factors$epi)
    n_nuc = nrow(res$Bparam$factors$nuc)
    n_clu = nrow(res$Bparam$factors$clu)

    T0 = stack(res$Bparam$T0, res$Bparam$factors$bt, res$Bparam$factors$br)
    y_tr = Y$sum(dim=c(3,4,5,6,7))

    m__ = make_m__(Y)
    f = factors_to_F(factors = res$Bparam$factors, missing_rate = m__)
    F_tensor = torch_diag_embed(f)

    if (method == "STRAND"){
        if ("lambda" %in% names(res$VIparam)){
            lam = res$VIparam$lambda
            lam = torch_cat(c(lam, torch_zeros(1,ncol(lam), device=device)), dim=1)
            theta = nnf_softmax(lam, dim=1)
        } else if("Lambda" %in% names(res$VIparam)) {
            theta = res$VIparam$Lambda/(res$VIparam$Lambda$sum(dim=1, keepdim=TRUE))
        }
    } else if(method == "TENSIG"){
        X = torch_cat(c(X, torch_ones(1, nrow(X), device=device)), dim=1)
        mu = res$VIparam$Gamma$matmul(X$transpose(1,2))
        eta = torch_cat(c(mu, torch_zeros(1,ncol(mu), device=device)), dim=1)
        theta = nnf_softmax(eta, dim=1)
    }

    pred = T0$matmul(F_tensor)$matmul(theta*Y$sum(dim=c(1,2,3,4,5,6)))

    dev = 2*(Y*torch_log(Y/(pred+1e-10)+1e-10) -Y+pred)$sum()

    return(dev)
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
estimateEffect <- function(mod0, X, niter=1000, sig_number, to_plot=TRUE){
    ## Simulate signature proportion from model
    sim_estimates =matrix(0, nr= niter, ncol = ncol(X))
    for(i in 1:niter){
        mu_tmp = as_array(mod0$VIparam$lambda$cpu())
        l_tmp = matrix(0, nr=nrow(mu_tmp), nc=ncol(mu_tmp))
        for(j in 1:ncol(mu_tmp)){
            l_tmp[,j] = rmvnorm(n=1,mean = mu_tmp[,j], sigma = round(sqrt(as_array(mod0$VIparam$Delta[j]$cpu())),2))
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
        covar_names = paste(colnames(X[,-1]),collapse="+")
        tmp = summary(lm(as.formula(paste0(sig_name, "~",covar_names)), data= df_tmp))
        sim_estimates[i,] = coef(tmp)[,"Estimate"]

    }
    colnames(sim_estimates) = colnames(X)
    
    ## compute p-value
    est <- colMeans(sim_estimates)
    se <- sqrt(apply(sim_estimates,2, stats::var))
    tval <- est/se
    rdf = nrow(X) - length(est)
    p_val <- 2*stats::pt(abs(tval), rdf, lower.tail=FALSE)
    coef_summary = cbind(est,se, tval, p_val)
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

