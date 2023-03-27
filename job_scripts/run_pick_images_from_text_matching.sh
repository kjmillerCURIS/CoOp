#!/bin/bash -l

#$ -P ivc-ml
#$ -pe omp 1
#$ -l h_rt=2:59:59
#$ -N pick_images_from_text_matching
#$ -j y
#$ -m ea

module load miniconda
conda activate dassl42
cd ~/data/CoOp
python pick_images_from_text_matching.py ../vislang-domain-exploration-data/CoCoOpExperiments/EDA/laion_class_text_matching ../vislang-domain-exploration-data/LAION-5B-Subset/ImageEmbeddingsAndMetadata

