#!/bin/bash -l

#$ -P ivc-ml
#$ -pe omp 2
#$ -l h_rt=01:59:59
#$ -N try_clip_features_text_on_holdout
#$ -j y
#$ -m ea

module load miniconda
conda activate dassl42
cd ~/data/CoOp
python try_clip_features_text_on_holdout.py

