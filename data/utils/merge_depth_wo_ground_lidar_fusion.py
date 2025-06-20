import os
import torch
import open3d as o3d
import json
from imageio.v2 import imread
import numpy as np
import cv2
from tqdm import tqdm
import argparse


def visualize_depth(depth, title="Depth Map"):
    """
    可视化深度图
    Args:
        depth (np.ndarray): 深度图数组
        title (str): 窗口标题
    """
    import matplotlib.pyplot as plt

    plt.figure(figsize=(10, 6))

    # 方法1：直接显示（适合相对深度）
    plt.subplot(1, 2, 1)
    plt.imshow(depth, cmap='gray')
    plt.title(f"{title} (Raw)")
    plt.colorbar()

    # 方法2：对数变换（增强暗部细节）
    plt.subplot(1, 2, 2)
    depth_log = np.log(depth + 1e-6)  # 避免log(0)
    plt.imshow(depth_log, cmap='jet')
    plt.title(f"{title} (Log Scale)")
    plt.colorbar()

    plt.tight_layout()
    plt.show()

def compare_depths(est_depth, lidar_depth, title_prefix=""):
    """
    对比估计深度与LiDAR深度
    Args:
        est_depth (np.ndarray): 估计深度图 (H,W)
        lidar_depth (np.ndarray): LiDAR深度图 (已reshape为H*W)
        title_prefix (str): 标题前缀
    """
    import matplotlib.pyplot as plt
    
    # 准备数据
    lidar_depth_2d = lidar_depth.reshape(est_depth.shape)
    valid_lidar_mask = lidar_depth > 0
    diff = np.abs(est_depth.reshape(-1)[valid_lidar_mask] - lidar_depth[valid_lidar_mask])
    
    # 创建4个子图
    plt.figure(figsize=(16, 10))
    
    # 子图1: 估计深度
    plt.subplot(2, 2, 1)
    plt.imshow(est_depth, cmap='jet', vmax=np.percentile(est_depth, 95))
    plt.colorbar(fraction=0.046, pad=0.04)
    plt.title(f"{title_prefix}Estimated Depth")
    
    # 子图2: LiDAR深度
    plt.subplot(2, 2, 2)
    lidar_vis = lidar_depth_2d.copy()
    lidar_vis[lidar_depth_2d <= 0] = np.nan  # 将无效值设为nan
    plt.imshow(lidar_vis, cmap='jet', vmax=np.percentile(lidar_depth[valid_lidar_mask], 95))
    plt.colorbar(fraction=0.046, pad=0.04)
    plt.title(f"{title_prefix}LiDAR Depth (Valid {valid_lidar_mask.mean():.1%})")
    
    # 子图3: 深度差异热力图
    plt.subplot(2, 2, 3)
    diff_map = np.abs(est_depth - lidar_depth_2d)
    diff_map[lidar_depth_2d <= 0] = 0  # 无LiDAR数据处设差异为0
    plt.imshow(diff_map, cmap='hot', vmax=np.percentile(diff, 95))
    plt.colorbar(fraction=0.046, pad=0.04)
    plt.title(f"Absolute Difference (Mean: {diff.mean():.2f}m)")
    
    # 子图4: 深度值分布直方图
    plt.subplot(2, 2, 4)
    plt.hist(est_depth.reshape(-1)[valid_lidar_mask], bins=50, alpha=0.5, 
             label='Estimated', range=(0, np.percentile(est_depth, 99)))
    plt.hist(lidar_depth[valid_lidar_mask], bins=50, alpha=0.5, 
             label='LiDAR', range=(0, np.percentile(lidar_depth[valid_lidar_mask], 99)))
    plt.legend()
    plt.title("Depth Distribution (Valid Pixels Only)")
    plt.xlabel("Depth (m)")
    
    plt.tight_layout()
    plt.show()



def save_point_cloud(points, colors, out_path, verbose=True):
    """Save 3D point cloud to file.

    Args:
        points (np.ndarray): Nx3 array of 3D point coordinates
        colors (np.ndarray): Nx3 array of RGB colors (0-1 range)
        out_path (str): Output file path
        verbose (bool): Whether to print save confirmation
    """
    try:
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.colors = o3d.utility.Vector3dVector(colors)
        o3d.io.write_point_cloud(out_path, pcd)
        if verbose:
            print(f"Saved point cloud to {out_path}")
    except Exception as e:
        print(f"Failed to save point cloud: {e}")
        raise


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
        intrinsic = np.array(frame["intrinsics"])
        c2w = np.array(frame["camtoworld"])

        cx, cy, fx, fy = (
            intrinsic[0, 2],
            intrinsic[1, 2],
            intrinsic[0, 0],
            intrinsic[1, 1],
        )
        H, W = frame["height"], frame["width"]

        rgb_path = frame["rgb_path"]
        frame_cam = frame["rgb_path"].split("/")[-2]
        im = np.array(imread(os.path.join(args.out, rgb_path)))
        depth_path = os.path.join(
            args.out,
            rgb_path.replace("images", "depth")
            .replace("./", "")
            .replace(".jpg", ".pt")
            .replace(".png", ".pt"),
        )
        depth = torch.load(depth_path).numpy()
        # visualize_depth(depth, f"Depth Frame: {frame_cam}")

        lidar_depth_path = os.path.join(
            args.out,
            rgb_path.replace("images", "lidar_depth")
            .replace("./", "")
            .replace(".jpg", ".npy")
            .replace(".png", ".npy"),
        )
        lidar_depth = np.load(lidar_depth_path).reshape(-1)

        # compare_depths(depth, lidar_depth, f"Frame {frame_cam} - ")

        x = np.arange(0, depth.shape[1])  # generate pixel coordinates
        y = np.arange(0, depth.shape[0])
        xx, yy = np.meshgrid(x, y)
        pixels = np.vstack((xx.ravel(), yy.ravel())).T.reshape(-1, 2)

        # unproject depth to pointcloud
        x = (pixels[..., 0] - cx) * depth.reshape(-1) / fx
        y = (pixels[..., 1] - cy) * depth.reshape(-1) / fy
        z = np.where(lidar_depth > 0, lidar_depth, depth.reshape(-1))
        local_points = np.stack([x, y, z], axis=1)
        local_colors = im.reshape(-1, 3).astype(np.float32) / 255.0

        # save_point_cloud(local_points, local_colors, os.path.join(
        #     args.out, "points3d_fusion_unsample.ply"))

        # z_infer = depth.reshape(-1)
        # local_points_infer = np.stack([x, y, z_infer], axis=1)
        # save_point_cloud(local_points_infer, local_colors, os.path.join(
        #     args.out, "points3d_infer_unsample.ply"))

        # valid_mask = lidar_depth > 0  # 创建有效点掩码
        # local_points_lidar = np.stack(
        #     [x[valid_mask], y[valid_mask], lidar_depth[valid_mask]], axis=1)
        # local_colors_filtered = local_colors[valid_mask]  # 过滤无效点对应的颜色
        # save_point_cloud(local_points_lidar, local_colors_filtered, os.path.join(
        #     args.out, "points3d_lidar_unsample.ply"))

        # mask dynamic
        mask_path = os.path.join(args.out,
                                 rgb_path.replace('images', 'masks').replace('./', '').replace('.jpg', '.npy').replace('.png', '.npy'))
        if os.path.exists(mask_path):
            dynamic_mask = np.load(mask_path).reshape(-1)

        # non-ground semantics
        smts_path = os.path.join(
            args.out,
            rgb_path.replace("images", "semantics")
            .replace("./", "")
            .replace(".jpg", ".npy")
            .replace(".png", ".npy"),
        )
        if os.path.exists(smts_path):
            smts = np.load(smts_path).reshape(-1)
            smt_mask = smts > 1

        mask = dynamic_mask & smt_mask
        # mask = smt_mask
        local_points = local_points[mask]
        local_colors = local_colors[mask]

        save_point_cloud(local_points, local_colors, os.path.join(
            args.out, "points3d_fusion_wo_ground_unsample.ply"))

        # random downsample
        if local_points.shape[0] < sample_per_frame:
            continue
        sample_idx = np.random.choice(
            np.arange(local_points.shape[0]), sample_per_frame
        )
        local_points = local_points[sample_idx]
        local_colors = local_colors[sample_idx]

        local_points_w = (c2w[:3, :3] @ local_points.T).T + c2w[:3, 3]

        save_point_cloud(local_points, local_colors, os.path.join(
            args.out, "points3d_fusion_sample.ply"))

        points.append(local_points_w)
        colors.append(local_colors)

    points = np.concatenate(points)
    colors = np.concatenate(colors)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    o3d.io.write_point_cloud(os.path.join(
        args.out, "points3d_fusion.ply"), pcd)
