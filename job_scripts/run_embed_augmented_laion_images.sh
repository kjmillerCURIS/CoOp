#!/bin/bash -l

#$ -P ivc-ml
#$ -pe omp 8
#$ -l gpus=1
#$ -l gpu_c=7.0
#$ -l h_rt=5:59:59
#$ -N embed_augmented_laion_images
#$ -j y
#$ -m ea

module load miniconda
conda activate dassl42
cd ~/data/CoOp
python embed_augmented_laion_images.py ../vislang-domain-exploration-data/CoCoOpExperiments/laion_data/uniform_subset

