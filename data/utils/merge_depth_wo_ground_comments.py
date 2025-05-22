# 导入必要的库
import os                # 文件路径操作
import torch             # PyTorch深度学习框架（用于加载深度图）
import open3d as o3d     # 点云处理库
import json              # JSON文件解析
from imageio.v2 import imread  # 读取图像文件（v2版本修复了某些兼容性问题）
import numpy as np       # 数值计算库
import cv2               # OpenCV图像处理库（虽然代码中未显式使用，可能为遗留依赖）
from tqdm import tqdm    # 进度条显示
import argparse           # 命令行参数解析

def get_opts():
    """解析命令行参数"""
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=str, required=True, help='输出目录路径')
    parser.add_argument('--total', type=int, default=1500000, help='总采样点数')
    return parser.parse_args()

if __name__ == "__main__":
    args = get_opts()  # 获取命令行参数
    
    # 读取元数据文件（包含相机参数、图像路径等信息）
    with open(os.path.join(args.out, 'meta_data.json'), 'r') as rf:
        meta_data = json.load(rf)

    ##########################################################################
    #                       像素反投影到3D点云处理流程                          #
    ##########################################################################
    
    points, colors = [], []  # 存储全局点云和颜色
    sample_per_frame = args.total // len(meta_data["frames"])  # 每帧采样点数
    
    # 遍历所有帧数据并显示进度条
    for frame in tqdm(meta_data["frames"]):
        # 获取相机内参矩阵和位姿矩阵
        intrinsic = np.array(frame["intrinsics"])  # 3x3相机内参矩阵
        c2w = np.array(frame["camtoworld"])        # 4x4相机到世界坐标变换矩阵

        # 解构内参参数
        cx, cy = intrinsic[0, 2], intrinsic[1, 2]  # 光心坐标
        fx, fy = intrinsic[0, 0], intrinsic[1, 1]  # 焦距
        H, W = frame["height"], frame["width"]     # 图像尺寸

        # 读取RGB图像
        rgb_path = frame["rgb_path"]
        im = np.array(imread(os.path.join(args.out, rgb_path)))

        # 构建深度图路径（根据图像路径转换）
        depth_path = os.path.join(
            args.out,
            rgb_path.replace("images", "depth")  # 替换目录
                   .replace("./", "")            # 移除相对路径符号
                   .replace(".jpg", ".pt")       # 替换扩展名
                   .replace(".png", ".pt"),
        )
        depth = torch.load(depth_path).numpy()  # 加载深度图（PyTorch tensor转numpy）

        # 生成像素坐标网格
        x = np.arange(0, depth.shape[1])  # 宽度方向坐标
        y = np.arange(0, depth.shape[0])  # 高度方向坐标
        xx, yy = np.meshgrid(x, y)        # 生成网格坐标
        pixels = np.vstack((xx.ravel(), yy.ravel())).T.reshape(-1, 2)  # 展平为Nx2像素坐标数组

        # 深度反投影计算（将像素坐标转换为相机坐标系下的3D点）
        x = (pixels[..., 0] - cx) * depth.reshape(-1) / fx  # X = (u - cx) * Z / fx
        y = (pixels[..., 1] - cy) * depth.reshape(-1) / fy  # Y = (v - cy) * Z / fy
        z = depth.reshape(-1)                                # Z = depth
        local_points = np.stack([x, y, z], axis=1)          # 组合成Nx3点云数组
        
        # 提取对应颜色信息（归一化到[0,1]）
        local_colors = im.reshape(-1, 3).astype(np.float32) / 255.0

        # 加载动态物体掩码（如果存在）
        mask_path = os.path.join(args.out,
                                rgb_path.replace('images', 'masks')
                                       .replace('./', '')
                                       .replace('.jpg', '.npy')
                                       .replace('.png', '.npy'))
        if os.path.exists(mask_path):
            dynamic_mask = np.load(mask_path).reshape(-1)

        # 加载语义分割结果（排除地面类别）
        smts_path = os.path.join(
            args.out,
            rgb_path.replace("images", "semantics")
                   .replace("./", "")
                   .replace(".jpg", ".npy")
                   .replace(".png", ".npy"),
        )
        if os.path.exists(smts_path):
            smts = np.load(smts_path).reshape(-1)
            smt_mask = smts > 1  # 标签>1为非地面类别
            
        # 组合掩码（当前版本仅使用语义掩码）
        # mask = dynamic_mask & smt_mask  # 原被注释掉的联合掩码
        mask = smt_mask                   # 当前实际使用的掩码
        
        # 应用掩码过滤点和颜色
        local_points = local_points[mask]
        local_colors = local_colors[mask]

        # 随机下采样（控制每帧贡献的点数）
        if local_points.shape[0] < sample_per_frame:
            continue  # 跳过点数不足的帧
        sample_idx = np.random.choice(
            np.arange(local_points.shape[0]), sample_per_frame
        )
        local_points = local_points[sample_idx]
        local_colors = local_colors[sample_idx]

        # 将局部点云转换到世界坐标系
        local_points_w = (c2w[:3, :3] @ local_points.T).T + c2w[:3, 3]

        # 收集全局点云
        points.append(local_points_w)
        colors.append(local_colors)

    # 合并所有帧的点云数据
    points = np.concatenate(points)
    colors = np.concatenate(colors)

    # 创建并保存Open3D点云对象
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)  # 设置点坐标
    pcd.colors = o3d.utility.Vector3dVector(colors)  # 设置点颜色
    o3d.io.write_point_cloud(os.path.join(args.out, "points3d.ply"), pcd)  # 写入PLY文件
