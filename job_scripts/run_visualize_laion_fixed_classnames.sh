#!/bin/bash -l

#$ -P ivc-ml
#$ -pe omp 3
#$ -l h_rt=01:59:59
#$ -N visualize_laion_fixed_classnames
#$ -j y
#$ -m ea

module load miniconda
conda activate dassl42
cd ~/data/CoOp
python visualize_laion_fixed_classnames.py ../vislang-domain-exploration-data/CoCoOpExperiments/laion_data/uniform_subset 100 ../vislang-domain-exploration-data/CoCoOpExperiments/EDA/laion_fixed_classname_vis

