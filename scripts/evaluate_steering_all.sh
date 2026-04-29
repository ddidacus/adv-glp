#!/bin/bash

#echo "================ Evaluating steering: classic ================ "
bash scripts/evaluate_steering.sh configs/paper/steering/refusal_glp.yaml
bash scripts/evaluate_steering.sh configs/paper/steering/refusal_noglp.yaml

#echo "================ Evaluating steering: compliance_direct ================ "
#bash scripts/evaluate_steering.sh configs/paper/steering/compliance_direct.yaml

#echo "================ Evaluating steering: compliance_glp ================ "
bash scripts/evaluate_steering.sh configs/paper/steering/compliance_glp.yaml
bash scripts/evaluate_steering.sh configs/paper/steering/compliance_noglp.yaml
bash scripts/evaluate_steering.sh configs/paper/steering/nosteering_glp.yaml
bash scripts/evaluate_steering.sh configs/paper/steering/nosteering_noglp.yaml

#echo "================ Evaluating steering: glp ================ "
#bash scripts/evaluate_steering.sh configs/paper/steering/glp.yaml

#echo "================ Evaluating steering: random ================ "
#bash scripts/evaluate_steering.sh configs/paper/steering/random.yaml
