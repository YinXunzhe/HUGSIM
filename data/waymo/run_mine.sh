#!/bin/bash

export HUGSIM_WORKSPACE=$HOME/hugsim_workspace

cuda=0
export CUDA_VISIBLE_DEVICES=$cuda

# base_dir="/nas/datasets/Waymo_NOTR/static"
# segment="segment-10061305430875486848_1080_000_1100_000_with_camera_labels.tfrecord"

base_dir="${HUGSIM_WORKSPACE}/datasets/waymo_NOTR/"
segment="segment-12505030131868863688_1740_000_1760_000_with_camera_labels.tfrecord"

seg_prefix=$(echo $segment| cut -c 9-15)
seq_name=scene-${seg_prefix}
out=${HUGSIM_WORKSPACE}/hugsim_data/waymo/${seq_name}
cameras="1 2 3"


mkdir -p $out

cd ${HUGSIM_WORKSPACE}/HUGSIM/data

# load images, camera pose, etc
python waymo/load.py -b ${base_dir} -c ${cameras} -o ${out} -s ${segment}

# generate semantic mask
cd InverseForm
./infer_waymo.sh ${cuda} ${out}
cd -

python utils/create_dynamic_mask.py --data_path ${out} --data_type waymo
python utils/estimate_depth.py --out ${out}
python utils/merge_depth_wo_ground.py --out ${out} --total 200000
python utils/merge_depth_ground.py --out ${out} --total 200000 --datatype waymo