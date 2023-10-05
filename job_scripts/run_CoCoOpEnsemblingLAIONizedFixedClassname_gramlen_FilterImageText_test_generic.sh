#!/bin/bash -l

#$ -P ivc-ml
#$ -pe omp 5
#$ -l gpus=1
#$ -l gpu_c=7.0
#$ -l h_rt=5:59:59
#$ -j y
#$ -m ea

module load miniconda
conda activate dassl42
cd ~/data/CoOp
python train.py \
--root ../vislang-domain-exploration-data/DatasetsForCoCoOp \
--seed ${SEED} \
--dataset-config-file configs/datasets/domainnet_custom.yaml \
--config-file configs/trainers/custom/CoCoOpEnsemblingLAIONizedFixedClassname_gramlen${GRAMLEN}_FilterImageText_baseline.yaml \
--model-dir ${MODEL_DIR} \
--output-dir ${OUTPUT_DIR} \
--class-split-type ${CLASS_SPLIT_TYPE} \
--fewshot-seed ${FEWSHOT_SEED} \
--domain-split-index ${DOMAIN_SPLIT_INDEX} \
--load-epoch 10 \
--eval-only \
--eval-type ${EVAL_TYPE}

