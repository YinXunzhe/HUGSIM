import os
import torch
import open3d as o3d
import json
from imageio.v2 import imread
import numpy as np
import cv2
from tqdm import tqdm
import argparse


def get_opts():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=str, required=True)
    parser.add_argument('--total', type=int, default=1500000)
    return parser.parse_args()


if __name__ == "__main__":
    args = get_opts()
    with open(os.path.join(args.out, 'meta_data.json'), 'r') as rf:
        meta_data = json.load(rf)

    ##########################################################################
    #                        unproject pixels                             #
    ##########################################################################

    points, colors = [], []
    sample_per_frame = args.total // len(meta_data["frames"])
    for frame in tqdm(meta_data["frames"]):
        rgb_path = frame["rgb_path"]
        frame_cam = frame["rgb_path"].split("/")[-2]
        if (frame_cam != 'CAM_FRONT_120'):
            continue

        intrinsic = np.array(frame["intrinsics"])
        c2w = np.array(frame["camtoworld"])

        cx, cy, fx, fy = (
            intrinsic[0, 2],
            intrinsic[1, 2],
            intrinsic[0, 0],
            intrinsic[1, 1],
        )
        H, W = frame["height"], frame["width"]

        # 获取帧索引，从rgb_path中提取
        frame_idx = int(rgb_path.split("/")[-1].split(".")[0])
        
        # 构建对应的ply文件路径
        ply_path = os.path.join(args.out, "lidar_colored", f"{str(frame_idx).zfill(6)}.ply")
        
        # 检查文件是否存在
        if not os.path.exists(ply_path):
            print(f"Warning: PLY file not found: {ply_path}")
            continue
            
        # 读取ply文件
        pcd = o3d.io.read_point_cloud(ply_path)
        
        # 获取点云数据和颜色
        local_points = np.asarray(pcd.points)
        local_colors = np.asarray(pcd.colors)
        
        # 如果点云数量不足，跳过该帧
        if local_points.shape[0] < sample_per_frame:
            print(f"Warning: Not enough points in frame {frame_idx}: {local_points.shape[0]} < {sample_per_frame}")
            continue
            
        # 随机下采样
        sample_idx = np.random.choice(
            np.arange(local_points.shape[0]), sample_per_frame
        )
        
        # 应用下采样
        local_points = local_points[sample_idx]
        local_colors = local_colors[sample_idx]

        local_points_w = (c2w[:3, :3] @ local_points.T).T + c2w[:3, 3]

        points.append(local_points_w)
        colors.append(local_colors)

    points = np.concatenate(points)
    colors = np.concatenate(colors)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    o3d.io.write_point_cloud(os.path.join(args.out, "points3d_lidar.ply"), pcd)