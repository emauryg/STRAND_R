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

    T_tensor[T_tensor < eps] = eps
    F_tensor[F_tensor < eps] = eps

    D = ncol(lam)
    if (sam_covs){
        lam = torch_cat(c(lam, torch_zeros(1, D, device=device)), dim=1)
    } else {
        lam = torch_log(lam + eps)
    }

    lam = lam$transpose(1,2)

    ## TODO: check if adding the 1e-14 is necessary 
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
#' @param res is the output of running \function{run_EM}. 
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

