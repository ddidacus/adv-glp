

echo "================ Evaluating: linear probe ================ "
bash eval_linear_probe.sh configs/paper/eval_lp.yaml

echo "================ Evaluating: GLP-PI ================ "
bash eval_classifier.sh configs/paper/eval_pi.yaml

echo "================ Evaluating: DTE ================ "
bash eval_classifier.sh configs/paper/eval_dte.yaml

echo "================ Evaluating: GLP-DTE ================ "
bash eval_classifier.sh configs/paper/eval_dte_glp.yaml

echo "================ Evaluating: DiffMean ================ "
bash eval_diffmean.sh configs/paper/eval_diffmean.yaml