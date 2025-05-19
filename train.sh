#!/bin/bash

export HUGSIM_WORKSPACE=$HOME/hugsim_workspace

seq_name=scene-1250503
input_path=${HUGSIM_WORKSPACE}/hugsim_data/waymo/$seq_name
output_path=${input_path}/outputs
dataset_name=waymo
mkdir -p ${output_path}

cd ${HUGSIM_WORKSPACE}/HUGSIM

python -u train_ground.py --data_cfg ./configs/${dataset_name}.yaml \
      --source_path ${input_path} --model_path ${output_path}

python -u train.py --data_cfg ./configs/${dataset_name}.yaml \
      --source_path ${input_path} --model_path ${output_path}