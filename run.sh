#!/bin/bash

mkdir -p logs

mkdir -p save_models

for model in DECISION_TREE RANDOM_FOREST MLP SVM XGBOOST
do
    echo "python -u trainer.py  --arch=$model  --save-dir=save_$model |& tee -a log_$model"
    python -u trainer.py  --arch=$model  --save-dir=save_models/$model.dat 2>&1 | tee -a logs/log_$model
done