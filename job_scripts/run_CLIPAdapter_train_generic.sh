#!/bin/bash -l

#$ -P ivc-ml
#$ -pe omp 3
#$ -l gpus=1
#$ -l gpu_c=5.0
#$ -l h_rt=11:59:59
#$ -j y
#$ -m ea

module load miniconda
conda activate dassl42
cd ~/data/CoOp
python train.py \
--root ../vislang-domain-exploration-data/DatasetsForCoCoOp \
--seed ${SEED} \
--dataset-config-file configs/datasets/domainnet_custom.yaml \
--config-file configs/trainers/custom/CLIPAdapter_baseline.yaml \
--output-dir ${OUTPUT_DIR} \
--resume ${OUTPUT_DIR} \
--class-split-type ${CLASS_SPLIT_TYPE} \
--fewshot-seed ${FEWSHOT_SEED} \
--domain-split-index ${DOMAIN_SPLIT_INDEX}
