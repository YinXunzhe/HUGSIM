"""
HUGSIM数据集读取模块
处理3D场景数据加载，包含相机参数解析、点云读取、场景信息整合等功能
"""

import os
from typing import NamedTuple
import numpy as np
import json
from plyfile import PlyData, PlyElement  # PLY文件处理
from utils.sh_utils import SH2RGB         # 球谐函数工具
from scene.gaussian_model import BasicPointCloud  # 基础点云数据结构
import torch.nn.functional as F
from imageio.v2 import imread             # 图像读取
import torch

class CameraInfo(NamedTuple):
    """单视图相机信息容器
    Attributes:
        K: 相机内参矩阵 3x3
        c2w: 相机到世界坐标系的变换矩阵 4x4
        image: 图像像素数据 [H,W,3]
        image_path: 图像文件路径
        image_name: 图像唯一标识名
        width: 图像宽度
        height: 图像高度
        semantic2d: 2D语义分割图 [H,W]
        optical_image: 光流数据 [H,W,2]
        depth: 深度图张量 [H,W]
        mask: 物体掩码 [H,W]
        timestamp: 时间戳（用于时序数据）
        dynamics: 动态物体参数（实例级运动信息）
    """
    K: np.array
    c2w: np.array
    image: np.array
    image_path: str
    image_name: str
    width: int
    height: int
    semantic2d: np.array
    optical_image: np.array
    depth: torch.tensor
    mask: np.array
    timestamp: int
    dynamics: dict

class SceneInfo(NamedTuple):
    """场景信息整合容器
    Attributes:
        point_cloud: 基础点云数据
        train_cameras: 训练视图相机列表
        test_cameras: 测试视图相机列表
        nerf_normalization: NeRF归一化参数（场景包围球）
        ply_path: 点云PLY文件路径
        verts: 动态物体顶点轨迹数据
    """
    point_cloud: BasicPointCloud
    train_cameras: list
    test_cameras: list
    nerf_normalization: dict
    ply_path: str
    verts: dict

def getNerfppNorm(cam_info, data_type):
    """计算NeRF++场景归一化参数（待完善实现）
    当前返回固定半径10的包围球，后续应基于相机位置计算实际包围球
    """
    def get_center_and_diag(cam_centers):
        """辅助函数：计算相机中心点包围球直径"""
        cam_centers = np.hstack(cam_centers)
        avg_cam_center = np.mean(cam_centers, axis=1, keepdims=True)
        center = avg_cam_center
        dist = np.linalg.norm(cam_centers - center, axis=0, keepdims=True)
        diagonal = np.max(dist)
        return center.flatten(), diagonal

    # 收集所有相机的世界坐标系位置
    cam_centers = []
    for cam in cam_info:
        cam_centers.append(cam.c2w[:3, 3:4])

    radius = 10  # TODO: 应替换为实际计算值
    return {'radius': radius}

def fetchPly(path):
    """从PLY文件加载点云数据
    Returns:
        BasicPointCloud: 包含点坐标、颜色、法线的对象
    """
    plydata = PlyData.read(path)
    vertices = plydata['vertex']
    positions = np.vstack([vertices['x'], vertices['y'], vertices['z']]).T
    
    # 处理颜色信息（优先使用PLY中的颜色通道）
    if 'red' in vertices:
        colors = np.vstack([vertices['red'], vertices['green'], vertices['blue']]).T / 255.0
    else:
        print('创建随机颜色')
        shs = np.ones((positions.shape[0], 3)) * 0.5
        colors = SH2RGB(shs)  # 使用球谐函数生成基础颜色
    
    # 初始化法线（当前为零向量）
    normals = np.zeros((positions.shape[0], 3))
    return BasicPointCloud(points=positions, colors=colors, normals=normals)

def storePly(path, xyz, rgb):
    """存储点云数据为PLY文件
    Args:
        path: 输出路径
        xyz: 点坐标数组 [N,3]
        rgb: 颜色数组 [N,3] 值域0-255
    """
    dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
            ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
            ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]
    
    normals = np.zeros_like(xyz)  # 法线暂存为零
    
    # 构建结构化数组
    elements = np.empty(xyz.shape[0], dtype=dtype)
    attributes = np.concatenate((xyz, normals, rgb), axis=1)
    elements[:] = list(map(tuple, attributes))
    
    # 写入文件
    vertex_element = PlyElement.describe(elements, 'vertex')
    ply_data = PlyData([vertex_element])
    ply_data.write(path)

def readHUGSIMCameras(path, data_type, ignore_dynamic):
    """读取HUGSIM数据集相机信息
    Args:
        path: 数据集根路径
        data_type: 数据集类型（kitti360/kitti等）
        ignore_dynamic: 是否忽略动态物体数据
    Returns:
        train_cam_infos: 训练相机列表
        test_cam_infos: 测试相机列表
        verts: 动态物体顶点轨迹
    """
    train_cam_infos, test_cam_infos = [], []
    with open(os.path.join(path, 'meta_data.json')) as json_file:
        meta_data = json.load(json_file)

        # 加载动态物体顶点轨迹
        verts = {}
        if 'verts' in meta_data and not ignore_dynamic:
            verts_list = meta_data['verts']
            for k, v in verts_list.items():
                verts[k] = np.array(v)

        # 处理每帧数据
        frames = meta_data['frames']
        for idx, frame in enumerate(frames):
            c2w = np.array(frame['camtoworld'])  # 相机外参矩阵
            
            # 构建图像路径并读取
            rgb_path = os.path.join(path, frame['rgb_path'].replace('./', ''))
            rgb_split = rgb_path.split('/')
            image_name = '_'.join([rgb_split[-2], rgb_split[-1][:-4]])
            image = imread(rgb_path)

            # 加载语义分割图（将14、15类合并到13类）
            semantic_2d = None
            semantic_pth = rgb_path.replace("images", "semantics").replace('.png', '.npy').replace('.jpg', '.npy')
            if os.path.exists(semantic_pth):
                semantic_2d = np.load(semantic_pth)
                semantic_2d[(semantic_2d == 14) | (semantic_2d == 15)] = 13

            # 加载光流数据
            optical_image = None
            optical_path = rgb_path.replace("images", "flow").replace('.png', '_flow.npy').replace('.jpg', '_flow.npy')
            if os.path.exists(optical_path):
                optical_image = np.load(optical_path)

            # 加载深度图
            depth = None
            depth_path = rgb_path.replace("images", "depth").replace('.png', '.pt').replace('.jpg', '.pt')
            if os.path.exists(depth_path):
                depth = torch.load(depth_path, weights_only=True)

            # 加载物体掩码
            mask = None
            mask_path = rgb_path.replace("images", "masks").replace('.png', '.npy').replace('.jpg', '.npy')
            if os.path.exists(mask_path):
                mask = np.load(mask_path)

            timestamp = frame.get('timestamp', -1)  # 获取时间戳

            intrinsic = np.array(frame['intrinsics'])  # 相机内参
            
            # 加载动态物体参数
            dynamics = {}
            if 'dynamics' in frame and not ignore_dynamic:
                dynamics_list = frame['dynamics']
                for iid in dynamics_list.keys():
                    dynamics[iid] = torch.tensor(dynamics_list[iid]).cuda()
            
            # 构建相机信息对象
            cam_info = CameraInfo(
                K=intrinsic, c2w=c2w, image=np.array(image),
                image_path=rgb_path, image_name=image_name, 
                height=image.shape[0], width=image.shape[1],
                semantic2d=semantic_2d, optical_image=optical_image,
                depth=depth, mask=mask, timestamp=timestamp,
                dynamics=dynamics
            )
            
            # 根据数据集类型划分训练/测试集
            if data_type == 'kitti360':
                # KITTI-360: 前20帧全训练，后续每20帧取后4帧测试
                if idx < 20:
                    train_cam_infos.append(cam_info)
                elif idx % 20 < 16:
                    train_cam_infos.append(cam_info)
                elif idx % 20 >= 16:
                    test_cam_infos.append(cam_info)
            elif data_type == 'kitti':
                # KITTI: 前10帧和最后4帧训练，中间每隔4帧取第3帧测试
                if idx < 10 or idx >= len(frames) - 4:
                    train_cam_infos.append(cam_info)
                elif idx % 4 < 2:
                    train_cam_infos.append(cam_info)
                elif idx % 4 == 2:
                    test_cam_infos.append(cam_info)
            # 其他数据集划分策略...
            else:
                raise NotImplementedError(f"未实现的数据集类型: {data_type}")

    return train_cam_infos, test_cam_infos, verts

def readHUGSIMInfo(path, data_type, ignore_dynamic):
    """整合HUGSIM场景信息
    Returns:
        SceneInfo: 包含点云、相机列表、归一化参数等的场景对象
    """
    # 读取相机信息和动态顶点
    train_cam_infos, test_cam_infos, verts = readHUGSIMCameras(path, data_type, ignore_dynamic)
    
    print(f'加载完成: {len(train_cam_infos)} 训练相机, {len(test_cam_infos)} 测试相机')
    
    # 计算场景归一化参数
    nerf_normalization = getNerfppNorm(train_cam_infos, data_type)
    
    # 加载点云数据
    ply_path = os.path.join(path, "points3d.ply")
    if not os.path.exists(ply_path):
        raise FileNotFoundError("需要预先生成初始点云文件 points3d.ply")
    
    try:
        pcd = fetchPly(ply_path)
    except Exception as e:
        print('点云加载失败:', e)
        exit(1)

    # 打包场景信息
    scene_info = SceneInfo(
        point_cloud=pcd,
        train_cameras=train_cam_infos,
        test_cameras=test_cam_infos,
        nerf_normalization=nerf_normalization,
        ply_path=ply_path,
        verts=verts
    )
    return scene_info

# 场景加载类型注册字典
sceneLoadTypeCallbacks = {
    "HUGSIM": readHUGSIMInfo,  # 注册HUGSIM数据加载器
}
 