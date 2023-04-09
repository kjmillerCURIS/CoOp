#!/bin/bash -l

#$ -P ivc-ml
#$ -pe omp 5
#$ -l gpus=1
#$ -l gpu_c=7.0
#$ -l h_rt=23:59:59
#$ -N select_laion_classname_by_matching
#$ -j y
#$ -m ea

module load miniconda
conda activate dassl42
cd ~/data/CoOp
python select_laion_classname_by_matching.py ../vislang-domain-exploration-data/CoCoOpExperiments/laion_data/uniform_subset

