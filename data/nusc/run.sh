#!/bin/zsh

export PYTHONPATH="${PWD}:$PYTHONPATH"

cuda=0
data='/workspace/datasets/nuscenes/v1.0-mini'
version='interp_12Hz_trainval'

# seq_list=('scene-0411' 'scene-0064' 'scene-0038' 'scene-0013')
seq_list=('scene-0796' )
for seq in "${seq_list[@]}"; do
        echo $seq
        start=0
        end=180
        out=/workspace/hugsim_data/nusc/${seq}

        export CUDA_VISIBLE_DEVICES=$cuda

        # mkdir -p ${out}
        # python nusc/load.py --datapath ${data} --version ${version} --seq ${seq} --out ${out} \
        #         --start ${start} --end ${end} --downsample 2 --video

        # python utils/vis_bbox_2d.py --out ${out}
        
        # generate semantic mask
        # cd InverseForm
        # ./infer_nuscenes.sh ${cuda} ${out}
        # cd -

        # python utils/create_dynamic_mask.py --data_path ${out} --data_type nuscenes

        COLMAP sparse model
        rm -rf ${out}/colmap_sparse*
        rm ${out}/database.db*
        rm -rf ${out}/prior
        python nusc/prepare_colmap.py -i ${out}

        echo "convert model into ply format"
        colmap model_converter \
                --input_path ${out}/colmap_sparse_ba \
                --output_path ${out}/sparse_ba.ply \
                --output_type PLY                

        python colmap/update_campose.py --datapath ${out}
        python utils/vis_bbox_2d.py --out ${out}

        python utils/estimate_depth.py --out ${out}
        python utils/merge_depth_wo_ground.py --out ${out} --total 200000
        python utils/merge_depth_ground.py --out ${out} --total 200000 --datatype nuscenes
done