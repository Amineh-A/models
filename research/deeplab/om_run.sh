#!/bin/bash
#SBATCH -N 1
#SBATCH -c 2
#SBATCH --array=0
#SBATCH --job-name=deeplab
#SBATCH --mem=32GB
#SBATCH --gres=gpu:tesla-k80:1
#SBATCH -t 12:00:00
#SBATCH -D /om/user/amineh/exp/log/
#SBATCH --partition=cbmm

cd /om/user/amineh/models/research/deeplab
singularity exec -B /om:/om --nv /om/user/xboix/singularity/xboix-tensorflow1.14.simg \
./run_polar.sh

