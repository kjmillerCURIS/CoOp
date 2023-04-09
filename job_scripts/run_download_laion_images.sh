#!/bin/bash -l

#$ -P ivc-ml
#$ -pe omp 12
#$ -l h_rt=23:59:59
#$ -N download_laion_images
#$ -j y
#$ -m ea

module load miniconda
conda activate dassl42
cd ~/data/CoOp
python download_laion_images.py ../vislang-domain-exploration-data/CoCoOpExperiments/laion_data/uniform_subset

