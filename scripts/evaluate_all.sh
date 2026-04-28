#!/bin/bash

echo "================ Evaluating: linear probe ================ "
sbatch scripts/evaluate_linear_probe.sh configs/paper/eval_lp.yaml

echo "================ Evaluating: GLP-PI ================ "
sbatch scripts/evaluate_classifier.sh configs/paper/eval_pi.yaml

echo "================ Evaluating: DTE ================ "
sbatch scripts/evaluate_classifier.sh configs/paper/eval_dte.yaml

echo "================ Evaluating: GLP-DTE ================ "
sbatch scripts/evaluate_classifier.sh configs/paper/eval_dte_glp.yaml

echo "================ Evaluating: DiffMean ================ "
sbatch scripts/evaluate_diffmean.sh configs/paper/eval_diffmean.yaml