"""
功能：从多帧深度图生成带地面调整的3D点云，并生成车辆控制指令
数据集支持：nuscenes, pandaset, waymo, kitti360
"""

import os
import torch
import open3d as o3d  # 用于点云处理和可视化
import json
from imageio.v2 import imread
import numpy as np
import cv2
from tqdm import tqdm  # 进度条显示
import argparse
import pickle

def get_opts():
    """解析命令行参数"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=str, required=True, help="输出目录路径")
    parser.add_argument("--total", type=int, default=1500000, help="总采样点数")
    parser.add_argument("--datatype", type=str, default="nuscenes", help="数据集类型")
    return parser.parse_args()

if __name__ == "__main__":
    args = get_opts()
    
    # 读取元数据文件
    with open(os.path.join(args.out, "meta_data.json"), "r") as rf:
        meta_data = json.load(rf)

    ##########################################################################
    #                        像素反投影到3D空间                            #
    ##########################################################################
    
    points, colors = [], []  # 存储全局点云数据
    sample_per_frame = args.total // len(meta_data["frames"])  # 每帧采样数
    front_cam_poses = []  # 存储前向摄像头位姿

    # 遍历所有帧数据
    for frame in tqdm(meta_data["frames"]):
        # 获取相机参数
        intrinsic = np.array(frame["intrinsics"])  # 相机内参矩阵
        c2w = np.array(frame["camtoworld"])       # 相机到世界的变换矩阵

        # 根据数据集类型记录前向摄像头位姿
        if args.datatype == "nuscenes" and "/CAM_FRONT/" in frame["rgb_path"]:
            front_cam_poses.append(c2w)
        elif args.datatype == "pandaset" and "/front_camera/" in frame["rgb_path"]:
            front_cam_poses.append(c2w)
        elif args.datatype == "waymo" and "/cam_1/" in frame["rgb_path"]:
            front_cam_poses.append(c2w)
        elif args.datatype == "kitti360" and "/cam_0/" in frame["rgb_path"]:
            front_cam_poses.append(c2w)
        elif args.datatype not in ["nuscenes", "pandaset", "waymo", "kitti360"]:
            raise NotImplementedError("不支持的数据集类型")

        # 图像参数
        H, W = frame["height"], frame["width"]
        cx, cy, fx, fy = (  # 内参分解
            intrinsic[0, 2], intrinsic[1, 2],
            intrinsic[0, 0], intrinsic[1, 1]
        )

        # 加载RGB图像和深度图
        im = imread(os.path.join(args.out, frame["rgb_path"]))
        depth_path = frame["rgb_path"].replace("images", "depth") \
                                      .replace(".jpg", ".pt") \
                                      .replace(".png", ".pt")
        depth = torch.load(os.path.join(args.out, depth_path)).numpy()

        # 生成像素坐标网格
        x = np.arange(W)
        y = np.arange(H)
        xx, yy = np.meshgrid(x, y)
        pixels = np.vstack([xx.ravel(), yy.ravel()]).T  # (H*W, 2)

        # 反投影到相机坐标系
        x = (pixels[:, 0] - cx) * depth.ravel() / fx
        y = (pixels[:, 1] - cy) * depth.ravel() / fy
        z = depth.ravel()
        local_points = np.column_stack([x, y, z])  # (H*W, 3)

        # 应用地面语义过滤
        smts_path = frame["rgb_path"].replace("images", "semantics") \
                                     .replace(".jpg", ".npy") \
                                     .replace(".png", ".npy")
        if os.path.exists(smts_path):
            smts = np.load(smts_path).ravel()      # 加载语义分割结果
            mask = smts <= 1                       # 0-1为地面类别
            local_points = local_points[mask]
            local_colors = im.reshape(-1, 3)[mask] / 255.0
        else:
            local_colors = im.reshape(-1, 3) / 255.0

        # 随机下采样
        if len(local_points) >= sample_per_frame:
            idx = np.random.choice(len(local_points), sample_per_frame)
            local_points = local_points[idx]
            local_colors = local_colors[idx]

        # 转换到世界坐标系
        world_points = (c2w[:3, :3] @ local_points.T).T + c2w[:3, 3]
        
        points.append(world_points)
        colors.append(local_colors)

    # 合并所有帧数据
    points = np.concatenate(points)
    colors = np.concatenate(colors)

    ##########################################################################
    #                       多平面地面模型调整                             #
    ##########################################################################
    
    # 设置前向摄像头高度（不同数据集不同参数）
    if args.datatype == "kitti360":
        front_cam_height = 1.55
    elif args.datatype == 'pandaset':
        front_cam_height = 2.2
    else:
        with open(os.path.join(args.out, "front_info.json"), "r") as f:
            front_info = json.load(f)
            front_cam_height = front_info["height"]
            front_rect_mat = front_info["rect_mat"]  # 外参校正矩阵

    # 将所有帧中的前向摄像头位姿存入front_cam_poses数组，形成多摄像头位姿序列
    front_cam_poses = np.stack(front_cam_poses)  # (N, 4, 4)

    # 为每个地面点计算其与所有前向摄像头位置的距离，用于后续的最近邻摄像头选
    points_cam_dist = np.sqrt(
        ((points[:, None] - front_cam_poses[:-1, :3, 3])**2).sum(axis=-1)
    )
    nearest_cam_idx = np.argmin(points_cam_dist, axis=1)  # 最近摄像头索引

    # 坐标变换流程
    nearest_c2w = front_cam_poses[nearest_cam_idx]     # 最近摄像头的世界变换
    nearest_w2c = np.linalg.inv(nearest_c2w)           # 世界到相机变换
    
    # 将点转换到最近摄像头的坐标系
    points_local = (nearest_w2c[:, :3, :3] @ points.T).T + nearest_w2c[:, :3, 3]
    points_local[:, 1] = front_cam_height  # 调整Y坐标到地面高度
    
    # 转换回世界坐标系
    points = (nearest_c2w[:, :3, :3] @ points_local.T).T + nearest_c2w[:, :3, 3]

    # 创建并保存点云
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    o3d.io.write_point_cloud(os.path.join(args.out, "ground_points3d.ply"), pcd)

    ##########################################################################
    #                       生成高级控制指令                              #
    ##########################################################################
    
    # 预测未来帧的位置变化
    forecast = 20  # 预测步长
    threshold = 2.5  # 转向判断阈值
    high_level_commands = []
    
    for i in range(len(front_cam_poses)):
        # 获取当前和预测位姿
        if i + forecast < len(front_cam_poses):
            target_pose = front_cam_poses[i + forecast]
        else:
            target_pose = front_cam_poses[-1]
        
        # 计算相对运动
        rel_pose = np.linalg.inv(front_cam_poses[i]) @ target_pose
        x_offset = rel_pose[0, 3]  # X轴方向偏移
        
        # 生成控制指令
        if x_offset > threshold:
            high_level_commands.append(0)  # 右转
        elif x_offset < -threshold:
            high_level_commands.append(1)  # 左转
        else:
            high_level_commands.append(2)  # 直行

    # 保存参数
    with open(os.path.join(args.out, "ground_param.pkl"), "wb") as f:
        pickle.dump((front_cam_poses, front_cam_height, high_level_commands), f)
