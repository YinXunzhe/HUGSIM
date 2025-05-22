import pickle
import numpy as np

ground_param_path="/home/sczone/hugsim_workspace/hugsim_data/waymo/scene-1250503/ground_param.pkl"
with open(ground_param_path, "rb") as f:
    poses, height, cmds = pickle.load(f)
    
print("摄像头高度:", height)
print("控制指令示例:", cmds[:5]) 
print("位姿矩阵形状:", poses.shape)
print("首个位姿矩阵:\n", poses[0])
