#!/bin/bash -l

#$ -P ivc-ml
#$ -pe omp 3
#$ -l h_rt=01:59:59
#$ -N try_clip_features_to_detect_text
#$ -j y
#$ -m ea

module load miniconda
conda activate dassl42
cd ~/data/CoOp
python try_clip_features_to_detect_text.py

