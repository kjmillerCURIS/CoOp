#!/bin/bash -l

#$ -P ivc-ml
#$ -pe omp 12
#$ -l h_rt=2:59:59
#$ -N download_class_text_matching_images
#$ -j y
#$ -m ea

module load miniconda
conda activate dassl42
cd ~/data/CoOp
python download_class_text_matching_images.py ../vislang-domain-exploration-data/CoCoOpExperiments/EDA/laion_class_text_matching

