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

def get_Sphere_Norm(xyz):
    # from lib.config import cfg
    xyz_max = np.max(xyz, axis=0)
    xyz_min = np.min(xyz, axis=0)
    center = (xyz_max + xyz_min) / 2
    radius = np.linalg.norm(xyz_max - xyz_min) / 2.
    # scale = cfg.data.get('sphere_scale', 1.0)
    scale = 1.0
    radius *= scale
    
    return {
        'radius': radius, 
        'center': center,
    }

def get_opts():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=str, required=True)
    parser.add_argument('--total', type=int, default=1500000)
    return parser.parse_args()


if __name__ == "__main__":
    args = get_opts()
    dataset = args.out
    meta_data_pth = os.path.join(dataset, 'meta_data.json')

    with open(meta_data_pth, 'r') as f:
        meta_data = json.load(f)

    ##########################################################################
    #                        merge lidar and colmap points                           #
    ##########################################################################
    # 构建对应的ply文件路径
    colmap_ply_path = os.path.join(args.out, "sparse_ba.ply")
    lidar_ply_path = os.path.join(args.out, "points3d_lidar_wo_ground.ply")

    # 初始化点云数据
    points = []
    colors = []

    # 检查并读取lidar点云
    if os.path.exists(lidar_ply_path):
        lidar_pcd = o3d.io.read_point_cloud(lidar_ply_path)
        lidar_points = np.asarray(lidar_pcd.points)
        lidar_colors = np.asarray(lidar_pcd.colors)
        
        # Get sphere center and radius
        lidar_sphere_normalization = get_Sphere_Norm(lidar_points)
        sphere_center = lidar_sphere_normalization['center']
        sphere_radius = lidar_sphere_normalization['radius']

        if len(lidar_points) > 0:
            points.extend(lidar_points)
            colors.extend(lidar_colors)
        else:
            print(f"Error: Lidar point cloud is empty: {lidar_ply_path}")
            exit(1)

    else:
        print(f"Error: Lidar PLY file not found: {lidar_ply_path}")
        exit(1)

    # 检查并读取colmap点云
    if os.path.exists(colmap_ply_path):
        colmap_pcd = o3d.io.read_point_cloud(colmap_ply_path)
        colmap_points = np.asarray(colmap_pcd.points)
        colmap_colors = np.asarray(colmap_pcd.colors)
        
        # 排除相机附近和下方的点
        points_colmap_mask = np.ones(colmap_points.shape[0], dtype=np.bool_)
        extent=10
        frames = meta_data['frames']
        for frame in frames:
            c2w = np.array(frame['camtoworld'])
            camera_position =c2w[:3, 3]
            radius = np.linalg.norm(colmap_points - camera_position, axis=-1)
            mask = np.logical_or(radius < extent, colmap_points[:,1] > camera_position[1])
            points_colmap_mask = np.logical_and(points_colmap_mask, ~mask)        
        colmap_points = colmap_points[points_colmap_mask]
        colmap_colors = colmap_colors[points_colmap_mask]

        # 保留半径2倍于场景球体范围内的点
        points_colmap_dist = np.linalg.norm(colmap_points - sphere_center, axis=-1)
        mask = points_colmap_dist < 2 * sphere_radius
        points_colmap_xyz = colmap_points[mask]
        points_colmap_rgb = colmap_colors[mask]

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
        o3d.io.write_point_cloud(os.path.join(
            args.out, "points3d.ply"), pcd)        
    else:
        print("Error: No point cloud data available to merge")
