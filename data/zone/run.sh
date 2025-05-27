#!/bin/bash

export HUGSIM_WORKSPACE=$HOME/hugsim_workspace

cuda=0
export CUDA_VISIBLE_DEVICES=$cuda

# base_dir="/home/sczone/hugsim_workspace/datasets/zone/3D_data_LSJWK4095NS119733"
# segment="20250304_154844_0"

base_dir="/mnt/e2e-data/3D_data_LSJWK4095NS119733/"
segment="20250317_161633_1"
# segment="20250304_154844_2"

track_seq_id="1"

# seg_prefix=$(echo $segment| cut -c 9-15)
seg_prefix=$segment
seq_name=scene-${seg_prefix}
out=${HUGSIM_WORKSPACE}/hugsim_data/zone/${seq_name}

# cameras=(
#     "CAM_FRONT",
#     "CAM_FRONT_LEFT",
#     "CAM_FRONT_RIGHT",
#     "CAM_BACK",
#     "CAM_BACK_LEFT",
#     "CAM_BACK_RIGHT",
# )
cameras=("CAM_FRONT_120" "CAM_FRONT_LEFT" "CAM_FRONT_RIGHT")

mkdir -p $out

cd ${HUGSIM_WORKSPACE}/HUGSIM/data

# load images, camera pose, etc
# python zone/load.py -b ${base_dir} -c "${cameras[@]}" --downsample 2 -o ${out} -s ${segment} --track_seq_id ${track_seq_id}

# generate semantic mask
# cd InverseForm
# ./infer_zone.sh ${cuda} ${out}
# cd -

# python utils/create_dynamic_mask.py --data_path ${out} --data_type zone
# python utils/estimate_depth.py --out ${out}
python utils/merge_depth_wo_ground.py --out ${out} --total 200000
python utils/merge_depth_ground.py --out ${out} --total 200000 --datatype zone