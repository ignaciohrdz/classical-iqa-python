#!/usr/bin/env bash

# If your system is powerful enough, you can add "kadid10k" and "koniq10k"
datasets=("liveiqa" "csiq" "cidiq" "tid2013" "nitsiqa")
methods=("gmlog" "sseq")
regressors=("mlp" "svr")

path_data="${HOME}/projects/datasets/iqa_datasets"

for d in "${datasets[@]}"; do
  for m in "${methods[@]}"; do
    for r in "${regressors[@]}"; do
      echo "Experiment: ${m} (${r}) with ${d}"
      python train.py --dataset "${d}" --model "${m}" --path_datasets "${path_data}" --regressor "${r}" --overwrite
    done
  done
done

