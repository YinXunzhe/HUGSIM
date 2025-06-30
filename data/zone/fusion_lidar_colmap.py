import os
import torch
import open3d as o3d
import json
from imageio.v2 import imread
import numpy as np
import cv2
from tqdm import tqdm
import argparse
from sklearn import linear_model


def get_opts():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=str, required=True)
    parser.add_argument('--total', type=int, default=1500000)
    return parser.parse_args()


if __name__ == "__main__":
    args = get_opts()

    ##########################################################################
    #                        merge lidar and colmap points                           #
    ##########################################################################
    # 构建对应的ply文件路径
    colmap_ply_path = os.path.join(args.out, "sparse_ba_wo_ground.ply")
    lidar_ply_path = os.path.join(args.out, "points3d_lidar_wo_ground.ply")

    # 初始化点云数据
    points = []
    colors = []
    lidar_min_y = None  # 存储LiDAR点云的最高点（y最小值）

    # 检查并读取lidar点云
    if os.path.exists(lidar_ply_path):
        lidar_pcd = o3d.io.read_point_cloud(lidar_ply_path)
        lidar_points = np.asarray(lidar_pcd.points)
        lidar_colors = np.asarray(lidar_pcd.colors)
        
        # 计算LiDAR点云的最高点（y坐标最小值）
        if len(lidar_points) > 0:
            lidar_min_y = np.min(lidar_points[:, 1])
            points.extend(lidar_points)
            colors.extend(lidar_colors)
    else:
        print(f"Warning: Lidar PLY file not found: {lidar_ply_path}")

    # 检查并读取colmap点云
    if os.path.exists(colmap_ply_path):
        colmap_pcd = o3d.io.read_point_cloud(colmap_ply_path)
        colmap_points = np.asarray(colmap_pcd.points)
        colmap_colors = np.asarray(colmap_pcd.colors)
        
        # 筛选colmap中高于LiDAR的点
        if lidar_min_y is not None and len(colmap_points) > 0:
            # 注意：坐标系y垂直向下，所以y值越小表示高度越高
            above_mask = colmap_points[:, 1] < lidar_min_y
            above_points = colmap_points[above_mask]
            above_colors = colmap_colors[above_mask]
            
            points.extend(above_points)
            colors.extend(above_colors)
            print(f"Merged {len(above_points)} points above LiDAR height")
        else:
            # 如果没有LiDAR高度参考，合并全部colmap点云
            points.extend(colmap_points)
            colors.extend(colmap_colors)
    else:
        print(f"Warning: Colmap PLY file not found: {colmap_ply_path}")

    # 创建并保存合并后的点云
    if points:  # 确保有点云数据
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(np.array(points))
        pcd.colors = o3d.utility.Vector3dVector(np.array(colors))
        o3d.io.write_point_cloud(os.path.join(
            args.out, "points3d_lidar_colmap_fusion.ply"), pcd)
    else:
        print("Error: No point cloud data available to merge")
