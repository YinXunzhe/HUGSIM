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


def points_in_box_vectorized(points, box_center, box_size, box_rotation):
    """
    向量化判断多个点是否在3D边界框内

    Args:
        points: 点云坐标数组 (N, 3)
        box_center: 边界框中心点坐标 [x, y, z]
        box_size: 边界框尺寸 [length, height, width]
        box_rotation: 边界框旋转矩阵 (3x3)

    Returns:
        numpy.ndarray: 布尔数组，表示每个点是否在边界框内
    """
    # 将所有点转换到边界框坐标系
    points_centered = points - box_center  # 平移
    # 旋转变换 (等效于对每个点应用 box_rotation.T @ point_centered)
    points_local = np.dot(points_centered, box_rotation)

    # 计算每个点在各个维度上是否在边界框内
    half_size = box_size / 2
    in_box = (
        (np.abs(points_local[:, 0]) <= half_size[0]) &
        (np.abs(points_local[:, 1]) <= half_size[1]) &
        (np.abs(points_local[:, 2]) <= half_size[2])
    )

    return in_box


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
    ignore_dynamic = False

    with open(os.path.join(args.out, 'meta_data.json'), 'r') as rf:
        meta_data = json.load(rf)

    verts = {}
    if 'verts' in meta_data and not ignore_dynamic:
        verts_list = meta_data['verts']
        for k, v in verts_list.items():
            verts[k] = np.array(v)
    ##########################################################################
    #                        merge lidar points                             #
    ##########################################################################
    points, colors = [], []
    sample_per_frame = args.total // len(meta_data["frames"]) * 6
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
        ply_path = os.path.join(args.out, "lidar_colored",
                                f"{str(frame_idx).zfill(6)}.ply")

        # 检查文件是否存在
        if not os.path.exists(ply_path):
            print(f"Warning: PLY file not found: {ply_path}")
            continue

        # 读取ply文件
        pcd = o3d.io.read_point_cloud(ply_path)

        # 获取CAM_FRONT_120 相机坐标系下的点云数据和颜色
        local_points = np.asarray(pcd.points)
        local_colors = np.asarray(pcd.colors)

        # ransac筛选地面之上的点云
        local_points, local_colors = segment_ground_ransac(
            local_points, local_colors)

        # 基于高度保留地面之上的点云
        # print("Max y before filtering:", local_points[:, 1].max())
        # mask = local_points[:, 1] <= 1.45
        # local_points = local_points[mask]
        # local_colors = local_colors[mask] if local_colors is not None else None
        # print("Max y after filtering:", local_points[:, 1].max())

        # 去除打到ego的点
        ego_mask = (np.abs(local_points[:, 0]) < 1.5) & (
            local_points[:, 2] < 2.5) & (local_points[:, 2] > - 4.5)
        local_points = local_points[~ego_mask]
        local_colors = local_colors[~ego_mask]

        # 将点云从相机坐标系转换到世界坐标系
        local_points_w = (c2w[:3, :3] @ local_points.T).T + c2w[:3, 3]

        # 抠去动态物体
        points_to_keep = np.ones(len(local_points_w), dtype=bool)
        dynamics = {}
        if 'dynamics' in frame and not ignore_dynamic:
            for iid, rt in frame['dynamics'].items():
                vertices = verts[iid]
                lhw = np.array([
                    vertices[:, 0].ptp(),
                    vertices[:, 1].ptp(),
                    vertices[:, 2].ptp()
                ])
                # 获取物体到参考世界坐标系的位姿变换矩阵(b2w)
                b2w = np.array(rt)

                # 获取边界框中心点和旋转矩阵
                box_center = b2w[:3, 3]
                box_rotation = b2w[:3, :3]

                # 检查所有点是否在当前边界框内
                in_box = points_in_box_vectorized(
                    local_points_w, box_center, lhw, box_rotation)

                # 更新需要保留的点（排除在边界框内的点）
                points_to_keep = points_to_keep & (~in_box)

            local_points_w = local_points_w[points_to_keep]
            local_colors = local_colors[points_to_keep]

        # 如果点云数量不足，跳过该帧
        if local_points_w.shape[0] < sample_per_frame:
            print(
                f"Warning: Not enough points in frame {frame_idx}: {local_points_w.shape[0]} < {sample_per_frame}")
            continue

        # ## 方式1：随机下采样
        # sample_idx = np.random.choice(
        #     np.arange(local_points_w.shape[0]), sample_per_frame
        # )

        # # 应用下采样
        # local_points_w = local_points_w[sample_idx]
        # local_colors = local_colors[sample_idx]

        # 方式2：体素滤波
        temp_pcd = o3d.geometry.PointCloud()
        temp_pcd.points = o3d.utility.Vector3dVector(local_points_w)
        temp_pcd.colors = o3d.utility.Vector3dVector(local_colors)

        # 计算初始体素大小（基于目标采样数和点云包围盒）
        bounds = temp_pcd.get_axis_aligned_bounding_box()
        volume = bounds.volume()
        initial_voxel_size = (volume / sample_per_frame) ** (1/3)

        # 设置合理的点云数量范围（90%-110%的目标值）
        min_points = int(sample_per_frame * 0.9)
        max_points = int(sample_per_frame * 1.1)
        voxel_size = initial_voxel_size*0.4 
        # down_pcd = temp_pcd.voxel_down_sample(voxel_size)     

        down_pcd = None
        # 最多3次调整（利用立方反比关系快速收敛）
        for _ in range(3):
            down_pcd = temp_pcd.voxel_down_sample(voxel_size)
            current_points = len(down_pcd.points)
            
            if min_points <= current_points <= max_points:
                break
            
            # 利用立方反比关系计算新体素大小
            if current_points > 0:  # 避免除零错误
                # 计算比例因子 (N_current / N_target)^(1/3)
                ratio = (current_points / sample_per_frame) ** (1/3)
                voxel_size*= ratio
            else:
                # 特殊处理：采样后无点，大幅减小体素大小
                voxel_size *= 0.3


        # 获取下采样后的点云和颜色
        local_points_w = np.asarray(down_pcd.points)
        local_colors = np.asarray(down_pcd.colors)

        points.append(local_points_w)
        colors.append(local_colors)

    points = np.concatenate(points)
    colors = np.concatenate(colors)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    o3d.io.write_point_cloud(os.path.join(
        args.out, "points3d_lidar_wo_ground.ply"), pcd)
