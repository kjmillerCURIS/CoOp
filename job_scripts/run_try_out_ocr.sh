#!/bin/bash -l

#$ -P ivc-ml
#$ -l gpu_c=7.0
#$ -l gpus=1
#$ -pe omp 2
#$ -l h_rt=01:59:59
#$ -N try_out_ocr
#$ -j y
#$ -m ea

module load miniconda
conda activate dassl42
cd ~/data/CoOp
python try_out_ocr.py

