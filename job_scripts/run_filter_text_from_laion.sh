#!/bin/bash -l

#$ -P ivc-ml
#$ -pe omp 3
#$ -l h_rt=11:59:59
#$ -N filter_text_from_laion
#$ -j y
#$ -m ea

module load miniconda
conda activate dassl42
cd ~/data/CoOp
python filter_text_from_laion.py

