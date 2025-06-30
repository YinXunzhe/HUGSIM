#!/bin/bash

export HUGSIM_WORKSPACE=/workspace
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# seq_name=scene-20250317_161633_1
seq_name=scene-20250304_154844_2
dataset_name=zone

input_path=${HUGSIM_WORKSPACE}/hugsim_data/${dataset_name}/${seq_name}
output_path=${HUGSIM_WORKSPACE}/models/${dataset_name}/${seq_name}

mkdir -p ${output_path}

cd ${HUGSIM_WORKSPACE}/HUGSIM

python -u train_ground.py --base_cfg ./configs/${dataset_name}_gs_base.yaml --data_cfg ./configs/${dataset_name}.yaml \
      --source_path ${input_path} --model_path ${output_path}

python -u train.py --base_cfg ./configs/${dataset_name}_gs_base.yaml  --data_cfg ./configs/${dataset_name}.yaml \
      --source_path ${input_path} --model_path ${output_path}