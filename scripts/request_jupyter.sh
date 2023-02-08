#!/bin/bash
#SBATCH --job-name=jup_nb
#SBATCH --mem=16GB
#SBATCH --output=/om2/user/leokoz8/code/boa_rnn/results/slurm_out/%j.out

#SBATCH --qos=normal
#SBATCH --partition=normal

##SBATCH --qos=fietelab
##SBATCH --partition=fietelab

#SBATCH --gres=gpu:1
#SBATCH --time=0-12:00:00

cd /om2/user/leokoz8/code/boa_rnn
unset XDG_RUNTIME_DIR
#conda activate boa_rnn
jupyter notebook --ip=0.0.0.0 --port=9000 --no-browser --NotebookApp.token='' --NotebookApp.password=''