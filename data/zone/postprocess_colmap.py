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


def segment_ground_ransac(points, colors, max_distance=0.3, max_iterations=1000):
    """
    在相机坐标系中分割地面点云（地面是y轴最大的平面）
    参数:
        points: 点云坐标数组 (N,3)
        colors: 点云颜色数组 (N,3)
        max_distance: 点到平面的最大距离阈值
        max_iterations: RANSAC最大迭代次数
    返回:
        非地面点云和对应颜色
    """

    # 因为地面是y最大的平面，我们翻转y坐标使地面变为"最低"平面
    flipped_points = points.copy()
    flipped_points[:, 1] = -flipped_points[:, 1]  # 翻转y轴

    # 使用RANSAC拟合平面（现在地面是y最小的平面）
    ransac = linear_model.RANSACRegressor(
        linear_model.LinearRegression(),
        residual_threshold=max_distance,
        max_trials=max_iterations
    )

    # 使用X和Z坐标来预测Y坐标（拟合平面方程）
    X = flipped_points[:, [0, 2]]  # 使用X和Z坐标
    y = flipped_points[:, 1]       # 拟合Y坐标

    ransac.fit(X, y)

    # 地面点是inliers，我们需要非地面点
    non_ground_mask = ~ransac.inlier_mask_

    return points[non_ground_mask], colors[non_ground_mask]


def get_opts():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=str, required=True)
    parser.add_argument('--total', type=int, default=1500000)
    return parser.parse_args()


if __name__ == "__main__":
    args = get_opts()
    with open(os.path.join(args.out, 'meta_data.json'), 'r') as rf:
        meta_data = json.load(rf)

        ply_path = os.path.join(args.out, "sparse_ba.ply")

        # 检查文件是否存在
        if not os.path.exists(ply_path):
            print(f"Warning: PLY file not found: {ply_path}")

        # 读取ply文件
        pcd = o3d.io.read_point_cloud(ply_path)

        # 获取点云数据和颜色
        local_points = np.asarray(pcd.points)
        local_colors = np.asarray(pcd.colors)

        # 筛选地面之上的点云
        local_points, local_colors = segment_ground_ransac(
            local_points, local_colors)

        # 只保留地面之上的点云
        # print("Max y before filtering:", local_points[:, 1].max())
        # mask = local_points[:, 1] <= 1.45  
        # local_points = local_points[mask]
        # local_colors = local_colors[mask] if local_colors is not None else None
        # print("Max y after filtering:", local_points[:, 1].max())

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(local_points)
        pcd.colors = o3d.utility.Vector3dVector(local_colors)
        o3d.io.write_point_cloud(os.path.join(
            args.out, "sparse_ba_wo_ground.ply"), pcd)
