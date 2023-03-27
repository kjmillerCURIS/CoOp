#!/bin/bash -l

#$ -P ivc-ml
#$ -pe omp 1
#$ -l h_rt=23:59:59
#$ -j y
#$ -m ea

module load miniconda
conda activate dassl42
cd ~/data/CoOp
python do_class_text_matching.py ../vislang-domain-exploration-data/CoCoOpExperiments/EDA/laion_class_text_matching ../vislang-domain-exploration-data/LAION-5B-Subset/ImageEmbeddingsAndMetadata ${START_INDEX} 16

