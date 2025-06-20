#!/bin/bash
export HUGSIM_WORKSPACE=/workspace
export NUPLAN_DATA_ROOT=/workspace/datasets/nuscenes/v1.0-mini

cd $HUGSIM_WORKSPACE/ASAP
mkdir -p out
# activate the conda env which should already have the dependencies installed for ASAP
conda run -n ASAP python -m sAP3D.nusc_annotation_generator \
  --data_path $NUPLAN_DATA_ROOT \
  --data_version v1.0-mini \
  --ann_frequency 12 \
  --ann_strategy interp