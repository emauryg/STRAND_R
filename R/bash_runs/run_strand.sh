#!/bin/bash 

## Example:
##  bash run_strand.sh TensorSignature_muts/strand_input/count_tensor.pt TensorSignature_muts/strand_input/X_tensor.pt TensorSignature_muts/strand_output/


module load gcc/6.2.0 cuda/10.2
GPU_TYPE="teslaV100"
NUMBER_OF_GPU=1

count_tensor=${1}
X=${2}
output_dir=${3}
init_path=${4}

sbatch -t 12:00:00 --mem=20G -n 1 -c 4 -p gpu --gres=gpu:${GPU_TYPE}:${NUMBER_OF_GPU} \
    --wrap="LD_PRELOAD=/n/data1/bch/genetics/lee/eam63/projects/admixsig/glibc-2.31/lib/libm.so.6 Rscript \
        main_strand.R -c ${count_tensor} -x ${X} -k 20 --tau 1 -o ${output_dir} --init_path ${4}"


