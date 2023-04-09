#!/bin/bash -l

#$ -P ivc-ml
#$ -pe omp 5
#$ -l h_rt=11:59:59
#$ -j y
#$ -m ea

module load miniconda
conda activate dassl42
cd ~/data/CoOp
python find_nearest_laion_texts.py ../vislang-domain-exploration-data/CoCoOpExperiments/EDA/nearest_laion_text ../vislang-domain-exploration-data/CoCoOpExperiments/baselines/CoCoOp 2 ${DOMAIN_SPLIT_INDEX}

