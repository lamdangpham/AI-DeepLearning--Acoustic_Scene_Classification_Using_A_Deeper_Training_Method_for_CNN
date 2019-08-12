#!/bin/sh
#SBATCH --nodes=1
#SBATCH --partition=gpu-p100
#SBATCH --ntasks-per-node=1
#SBATCH --job-name=run.sh
#SBATCH --mem=150GB
#SBATCH --output=./slu_%j.out
python step02_training.py
sc03_extract_test.py
sc04_extract_train.py
sc05_cre_train_batch.py


