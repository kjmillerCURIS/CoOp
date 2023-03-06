#!/bin/bash -l

#$ -P ivc-ml
#$ -pe omp 3
#$ -l gpus=1
#$ -l gpu_c=5.0
#$ -l h_rt=11:59:59
#$ -N CoCoOp_test_b2n_IN1k
#$ -j y
#$ -m ea

module load miniconda
conda activate dassl42
cd ~/data/CoOp
python train.py \
--root ../vislang-domain-exploration-data/DatasetsForCoCoOp \
--seed 1 \
--trainer CoCoOp \
--dataset-config-file configs/datasets/imagenet.yaml \
--config-file configs/trainers/CoCoOp/vit_b16_c4_ep10_batch1_ctxv1.yaml \
--output-dir ../vislang-domain-exploration-data/CoCoOpExperiments/trial_runs/CoCoOp_b2n_IN1k/test_new/imagenet/shots_16/CoCoOp/vit_b16_c4_ep10_batch1_ctxv1/seed1 \
--model-dir ../vislang-domain-exploration-data/CoCoOpExperiments/trial_runs/CoCoOp_b2n_IN1k/train_base/imagenet/shots_16/CoCoOp/vit_b16_c4_ep10_batch1_ctxv1/seed1 \
--load-epoch 10 \
--eval-only \
DATASET.NUM_SHOTS 16 \
DATASET.SUBSAMPLE_CLASSES new

