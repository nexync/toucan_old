#!/bin/bash
#$ -q gpu.q@@rtx -cwd
#$ -l h_rt=36:00:00,gpu=4
#$ -N ccot
 
# NOTE: after the launch encodings script

echo $(date +%F_%H-%M-%S)
 
cd ~/ccot

export C=~/ccot/configs/toucan_10.yaml
export GPUS=4
bash scripts/run_exp.sh