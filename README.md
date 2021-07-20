# STRAND_R
Structural TensoR Analsyis and Decomposition

## Installation

`library(devtools); install_github("emauryg/STRAND_R");`

## Quickstart
```
library(MCMCpack); library(mvtnorm); library(NMF); library(clue); library(nnet);

library(Matrix); library(torch)

if (cuda_is_available()) {
   device <<- torch_device("cuda:0")
} else {
   device <<- torch_device("cpu")
}

D=150 ## number of samples
K=5 ## number of latent factors (signatures)
V=96 ## number of trinucleotide contexts
p=10 ## number of sample covariates

set.seed(777)
torch_manual_seed(777)

truth_vals <- generate_data(V,K,D,p)

count_matrix = truth_vals$count_matrix
xmat = truth_vals$X

## Initialize 
init_pars <- NMFinit(count_matrix,xmat,K, max_iter=10000) 

## Fit model
mod0 = runEM(init_pars, count_matrix, X=xmat)

## Plot 96 trinucleotide context
make_96plot(mod0,model_type = "STRAND")

## Plot factors
factor_plots = plot_factors(mod0)

```

## GPU acceleration
Using `torch` STRAND can make use of GPUs if the current device has that infrastructure available. 

```
if (cuda_is_available()) {
   device <<- torch_device("cuda:0")
} else {
   device <<- torch_device("cpu")
}
```
This can provide significant speed gains compared to just using cpu. 

### TODO:

* Need to incorporate package names of none base R for ease of use. 
