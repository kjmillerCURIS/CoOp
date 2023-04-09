#!/bin/bash -l

#$ -P ivc-ml
#$ -pe omp 3
#$ -l gpus=1
#$ -l gpu_c=7.0
#$ -l h_rt=11:59:59
#$ -j y
#$ -m ea

module load miniconda
conda activate dassl42
cd ~/data/CoOp
python embed_all_laion_captions.py ../vislang-domain-exploration-data/CoCoOpExperiments/EDA/nearest_laion_text ../vislang-domain-exploration-data/LAION-5B-Subset/ImageEmbeddingsAndMetadata ${START_INDEX} 10

