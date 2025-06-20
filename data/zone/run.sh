#!/bin/bash

# export HUGSIM_WORKSPACE=$HOME/hugsim_workspace
export HUGSIM_WORKSPACE=/workspace
export PYTHONPATH="${PWD}:$PYTHONPATH"
cuda=0
export CUDA_VISIBLE_DEVICES=$cuda

# base_dir="/home/sczone/hugsim_workspace/datasets/zone/3D_data_LSJWK4095NS119733"
# segment="20250304_154844_0"

base_dir="/mnt/e2e-data/3D_data_LSJWK4095NS119733/"
# segment="20250317_161633_1"
segment="20250304_154844_2"

track_seq_id="1"

# seg_prefix=$(echo $segment| cut -c 9-15)
seg_prefix=$segment
seq_name=scene-${seg_prefix}
out=${HUGSIM_WORKSPACE}/hugsim_data/zone/${seq_name}

mkdir -p $out

cd ${HUGSIM_WORKSPACE}/HUGSIM/data

# load images, camera pose, etc
python zone/load.py -b ${base_dir} --downsample 2 -o ${out} -s ${segment} --track_seq_id ${track_seq_id}

# generate semantic mask
cd InverseForm
./infer_zone.sh ${cuda} ${out}
cd -

# # COLMAP sparse model
# rm -rf ${out}/colmap_sparse*
# rm ${out}/database.db*
# rm -rf ${out}/prior
# python zone/prepare_colmap.py -i ${out}

# echo "convert model into ply format"
# colmap model_converter \
#         --input_path ${out}/colmap_sparse_tri \
#         --output_path ${out}/sparse_tri.ply \
#         --output_type PLY      

# colmap model_converter \
#         --input_path ${out}/colmap_sparse_ba \
#         --output_path ${out}/sparse_ba.ply \
#         --output_type PLY                

# python zone/postprocess_colmap.py --out ${out}

# python colmap/update_campose.py --datapath ${out}
# python utils/vis_bbox_2d.py --out ${out}

python utils/create_dynamic_mask.py --data_path ${out} --data_type zone
python utils/estimate_depth.py --out ${out}
python utils/merge_depth_wo_ground.py --out ${out} --total 200000
python utils/merge_depth_ground.py --out ${out} --total 200000 --datatype zone