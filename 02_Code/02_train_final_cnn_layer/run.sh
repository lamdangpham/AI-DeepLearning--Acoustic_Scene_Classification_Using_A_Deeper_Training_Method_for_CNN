#!/bin/sh
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --job-name=run.sh
#SBATCH --mem=150GB
#SBATCH --output=./slu_%j.out
python step02_training.py
