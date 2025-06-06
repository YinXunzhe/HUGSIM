#!/bin/bash
cuda=$1
out=$2

echo $cuda
echo $out
export CUDA_VISIBLE_DEVICES=$cuda

arr=("CAM_FRONT_120"  "CAM_FRONT_LEFT"  "CAM_FRONT_RIGHT")
for cam in ${arr[@]}
do
    echo ${cam}
    torchrun --nproc_per_node=1 validation.py \
    --input_dir ${out}/images/${cam} \
    --output_dir ${out}/semantics/${cam} \
    --model_path /home/sczone/hugsim_workspace/HUGSIM/data/InverseForm/checkpoints/hrnet48_OCR_HMS_IF_checkpoint.pth \
    --arch "ocrnet.HRNet_Mscale" --hrnet_base "48" --has_edge True
    echo Done
done
