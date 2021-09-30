

## Load all required libraries
library(devtools);
install_github("emauryg/STRAND_R",ref="mini_batch");
library(MCMCpack); library(mvtnorm); library(NMF); library(clue); library(nnet); library(luz)

library(batch); library(Matrix); library(torch); library(tidyverse)

library(strandR)
library(torch)
library(tidyverse)
library(argparse)


## Set up cuda device if available
if (cuda_is_available()) {
   device <<- torch_device("cuda:0")
} else {
   device <<- torch_device("cpu")
}


parser <- ArgumentParser()

parser$add_argument("-c", "--count_tensor", 
                        help="Count tensor file path (.pt)", type="character")
parser$add_argument("-x","--x_covariates", help="X covariates file path (.pt)", type="character")
parser$add_argument("-k","--num_signatures", 
    type="integer",help="Number of signatures", default=3)
parser$add_argument("--tau", default=1, 
    type="integer",help="weight of tnf regulation")
parser$add_argument("-o","--output_dir", help="Output directory", default = "./", type="character")
parser$add_argument("--init_path", help="Path to initiated model", default= NULL,type="character")

args <- parser$parse_args()

## Check that files can be accessed
if (is.null(args$count_tensor) || !file.exists(args$count_tensor)){
    stop(sprintf("Specified count tensor does not exist"))
} else {
    count_matrix <- torch_load(args$count_tensor)$to(device=device)
}

if(is.null(args$x_covariates) || !file.exists(args$x_covariates)){
    stop(sprintf("Specified x covariates does not exist"))
} else {
    X_tensor <- torch_load(args$x_covariates)$to(device=device)
}

if(!is.null(args$num_signatures)){
    K <- as.numeric(args$num_signatures)
} else{
    stop("Please specify the number of signatures -k, --num_signatures")
}

if(is.null(args$init_path)){

    cat("==========================\n")
    cat("Initializing values....\n")
    cat("==========================\n")
    t1 = Sys.time()
    init_pars <- NMFinit(count_matrix, X_tensor, K=K, max_iter= 10000)
    t2 = Sys.time()
    cat("It took: ",difftime(t2,t1, units= "mins")," minutes to initialize. \n")
    ## Save initialization results
    init_path = paste0(args$output_dir, "/init/")
    system(paste0("mkdir -p ", init_path))
    for(n in names(init_pars)){
        if(n == "covs"){
            for (f in  names(init_pars$covs)){
                torch_save(init_pars$covs[[f]]$cpu(), path=paste0(init_path,f,"_init.pt"))
            }
        } else{
            torch_save(init_pars[[n]]$cpu(),path=paste0(init_path,n,"_init.pt"))
        }
    }

} else{
    cat("==========================\n")
    cat("Loading initialization values....\n")
    cat("==========================\n")
    init_path = paste0(args$init_path,"/")
    init_pars = list(T0=NULL,eta=NULL, covs=list(bt=NULL,br=NULL,epi=NULL,nuc=NULL,clu=NULL),Delta=NULL,
                 Xi=NULL,Sigma=NULL,gamma_sigma=NULL,zeta=NULL)
    for(n in names(init_pars)){
        if(n == "covs"){
            for (f in  names(init_pars$covs)){
                init_pars$covs[[f]] = torch_load(path=paste0(init_path,f,"_init.pt"))$to(device=device)
            }
        } else{
        init_pars[[n]]= torch_load(path=paste0(init_path,n,"_init.pt"))$to(device=device)
        }
    }
}

cat("=========================\n")
cat("Running Variational EM step...\n")
cat("=========================\n")
mod0 = runEM(init_pars, count_matrix, X=X_tensor, tau=args$tau)

## Save model results
model_path= paste0(args$output_dir, "/model_output/")
system(paste0("mkdir -p ", model_path))
for(n in names(mod0)){
    if(n == "VIparam"){
        for(i in names(mod0$VIparam)){
            torch_save(mod0$VIparam[[i]]$cpu(), path=paste0(model_path,i,"_model.pt"))
        }
    } else{
        for(i in names(mod0$Bparam)){
            if(i == "factors"){
                for(j in names(mod0$Bparam$factors)){
                    torch_save(mod0$Bparam$factors[[j]]$cpu(), path= paste0(model_path, j,"_model.pt"))
                }
            } else{
                torch_save(mod0$Bparam[[i]]$cpu(), path=paste0(model_path, i,"_model.pt"))
            }
        }
    }
}