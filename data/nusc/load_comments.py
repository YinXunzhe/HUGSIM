"""
NuScenes 数据集加载与预处理脚本
功能：从NuScenes数据集中提取指定场景的多传感器数据，处理动态物体，生成训练/可视化所需的数据格式
"""

# 导入标准库
import os
import json
import shutil
import argparse
from collections import defaultdict

# 导入数据处理相关库
import numpy as np
import cv2
import open3d as o3d
import torch
import mediapy as media
from tqdm import tqdm
from scipy.spatial.transform import Rotation as R

# NuScenes 官方工具包
from nuscenes.nuscenes import NuScenes

# 自定义模块
from nusc.utils import (AVAILABLE_CAMERAS, WLH_TO_LWH, ALLOWED_CLASSES,
                       _rotation_translation_to_pose, find_all_sample_data, find_all_sample,
                       get_vertices, point_in_bbox, frame_check, get_sample_pose, load_cam,
                       traj_dict_to_list, get_box)

def get_opts():
    """解析命令行参数"""
    parser = argparse.ArgumentParser()
    parser.add_argument('--datapath', type=str, required=True, 
                      help='NuScenes数据集根目录路径')
    parser.add_argument('--version', type=str, required=True,
                      help='数据集版本，例如v1.0-trainval')
    parser.add_argument('--seq', type=str, required=True,
                      help='要处理的场景序列名称')
    parser.add_argument('--out', type=str, required=True,
                      help='输出目录路径')
    parser.add_argument('--downsample', type=int, default=2,
                      help='图像下采样因子')
    parser.add_argument('--start', type=int, default=0,
                      help='起始帧索引')
    parser.add_argument('--end', type=int, default=-1,
                      help='结束帧索引（-1表示处理到末尾）')
    parser.add_argument('--video', action="store_true", default=False,
                      help='是否生成全景视频')
    return parser.parse_args()

if __name__ == "__main__":
    # 解析命令行参数并初始化输出目录
    args = get_opts()
    os.makedirs(args.out, exist_ok=True)
    
    # 初始化元数据结构（用于后续的NeRF训练等任务）
    meta_data = {
        "camera_model": "OPENCV",  # 相机模型类型
        'verts': {},              # 动态物体的3D顶点信息
        "frames": [],             # 每帧的传感器数据记录
    }

    # 初始化NuScenes数据集接口
    nusc = NuScenes(version=args.version, dataroot=args.datapath, verbose=True)
    
    ###################################################################
    #                   第一阶段：传感器初始化与地面估计                    #
    # 在首帧中计算各传感器的相对位姿，并估计地面平面方程                    #
    ###################################################################
    
    # 获取指定场景并定位第一个样本
    scene = nusc.get("scene", nusc.field2token("scene", "name", args.seq)[0])
    first_sample = nusc.get("sample", scene["first_sample_token"])

    # ---------------------- 激光雷达处理 --------------------------
    # 获取LIDAR_TOP的校准数据并计算其在世界坐标系中的位姿
    lidar_data = nusc.get('sample_data', first_sample["data"]["LIDAR_TOP"])
    calibrated_lidar = nusc.get("calibrated_sensor", lidar_data["calibrated_sensor_token"])
    ego_pose_data = nusc.get("ego_pose", lidar_data["ego_pose_token"])
    
    # 构建变换矩阵：lidar -> ego -> world
    ego_pose = _rotation_translation_to_pose(ego_pose_data["rotation"], ego_pose_data["translation"])
    lidar_pose = _rotation_translation_to_pose(calibrated_lidar["rotation"], calibrated_lidar["translation"])
    lidar2w = ego_pose @ lidar_pose  # 最终的lidar到世界变换矩阵

    # 地面平面估计（使用RANSAC算法）
    points = np.fromfile(os.path.join(args.datapath, lidar_data["filename"]), 
                       dtype=np.float32).reshape([-1, 5])[:, :3]
    # 创建地面和自车区域掩膜
    ground_mask = (np.abs(points[:, 0]) < 3) & (np.abs(points[:, 1]) < 6)
    ego_mask = (np.abs(points[:, 0]) < 1.5) & (np.abs(points[:, 1]) < 2.5)
    filtered_points = points[ground_mask & ~ego_mask]
    
    # 转换到lidar坐标系并进行平面拟合
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(filtered_points)
    plane_model, _ = pcd.segment_plane(distance_threshold=0.01, ransac_n=3, num_iterations=1000)
    a, b, c, d = plane_model  # 平面方程系数 ax + by + cz + d = 0
    o3d.io.write_point_cloud(os.path.join(args.out, 'ground_lidar.ply'), pcd)  # 保存地面点云

    # ---------------------- 多相机系统初始化 ------------------------
    all_campose = {}
    for cam in AVAILABLE_CAMERAS:
        # 获取每个相机的内外参信息
        cam_data = nusc.get('sample_data', first_sample["data"][cam])
        calibrated_cam = nusc.get("calibrated_sensor", cam_data["calibrated_sensor_token"])
        all_campose[cam] = _rotation_translation_to_pose(
            calibrated_cam["rotation"], calibrated_cam["translation"]
        )

    # 计算前向相机的离地高度和俯仰角修正矩阵
    front_cam_t = all_campose['CAM_FRONT'][:3, 3]
    height = -(a * front_cam_t[0] + b * front_cam_t[1] + d) / c  # 代入平面方程计算高度
    n = np.array([a, b, c])
    front_cam_z = all_campose['CAM_FRONT'][:3, 0]  # 相机Z轴方向（OpenCV坐标系）
    pitch_angle = np.arccos(np.dot(n, front_cam_z) / (np.linalg.norm(n) * np.linalg.norm(front_cam_z)))
    rect_pitch = np.pi / 2 - pitch_angle  # 计算需要修正的俯仰角
    rect_mat = R.from_euler('x', rect_pitch).as_matrix()  # 生成绕X轴的旋转矩阵
    
    # 保存前向相机关键参数
    front_cam_info = {
        "height": front_cam_t[2] - height,  # 绝对高度
        "rect_mat": rect_mat.tolist(),     # 姿态修正矩阵
    }
    with open(os.path.join(args.out, 'front_info.json'), 'w') as f:
        json.dump(front_cam_info, f)

    # -------------------- 多相机刚性配置生成 ------------------------
    # 用于后续的捆绑调整（Bundle Adjustment），建立相机间的相对位姿关系
    cam_rigid = {
        "ref_camera_id": 1,  # 以CAM_FRONT作为参考相机
        "cameras": []
    }
    ref_extrinsic = all_campose['CAM_FRONT']
    for iid, cam_name in enumerate(AVAILABLE_CAMERAS):
        # 计算每个相机相对于参考相机的变换
        rel_extrinsic = np.linalg.inv(all_campose[cam_name]) @ ref_extrinsic
        qvec = R.from_matrix(rel_extrinsic[:3, :3]).as_quat()  # 旋转矩阵转四元数
        
        cam_config = {
            "camera_id": iid + 1,
            "image_prefix": f'{cam_name}/',
            'cam_from_rig_rotation': [qvec[3], qvec[0], qvec[1], qvec[2]],  # 四元数(w, x, y, z)
            'cam_from_rig_translation': rel_extrinsic[:3, 3].tolist(),
        }
        cam_rigid["cameras"].append(cam_config)
    
    with open(os.path.join(args.out, "cam_rigid_config.json"), "w") as f:
        json.dump([cam_rigid], f, indent=4)

    ###################################################################
    #                   第二阶段：序列帧数据处理                         #
    # 处理整个序列的所有帧，提取图像、位姿、动态物体信息                   #
    ###################################################################
    
    # 获取所有样本并应用帧范围过滤
    samples = find_all_sample(nusc, first_sample)[args.start:args.end]
    
    # 计算初始帧的逆位姿（用于后续坐标归一化）
    fff_sample_data = nusc.get("sample_data", samples[0]['data']["CAM_FRONT"])
    inv_pose = np.linalg.inv(get_sample_pose(nusc, fff_sample_data)[0])
    meta_data['inv_pose'] = inv_pose.tolist()  # 保存到元数据中

    # 初始化图像输出目录
    shutil.rmtree(os.path.join(args.out, 'images'), ignore_errors=True)
    for cam in AVAILABLE_CAMERAS:
        os.makedirs(os.path.join(args.out, 'images', cam), exist_ok=True)

    # ------------------- 动态物体轨迹分析 ----------------------
    # 遍历所有样本，检测移动超过阈值的动态物体
    tracks = defaultdict(list)
    for sample in samples:
        for box_token in sample['anns']:
            instance_token, pose, _ = get_box(nusc, box_token, inv_pose)
            tracks[instance_token].append(pose)
    
    # 识别动态实例（位移超过2米）
    dynamic_instance = set()
    for instance_token, traj_list in tracks.items():
        if np.linalg.norm(traj_list[0][:3, 3] - traj_list[-1][:3, 3]) > 2:
            dynamic_instance.add(instance_token)

    # ------------------- 主处理循环 --------------------------
    video_images = []  # 视频帧缓存
    start_time = -1     # 时间戳基准
    
    for i, sample in tqdm(enumerate(samples), desc="Processing frames"):
        # 处理动态物体位姿
        dynamics = {}
        for box_token in sample["anns"]:
            instance_token, pose, lhw = get_box(nusc, box_token, inv_pose)
            if instance_token in dynamic_instance:
                dynamics[instance_token] = pose.tolist()
                # 首次出现时记录物体包围盒顶点
                if instance_token not in meta_data['verts']:
                    meta_data['verts'][instance_token] = get_vertices(lhw).tolist()

        # 处理多相机数据
        cat_images = []  # 当前帧的全景拼接图像
        for cam in AVAILABLE_CAMERAS:
            # 加载相机数据
            sample_data = nusc.get('sample_data', sample['data'][cam])
            im, im_name, height, width, intrinsic, pose = load_cam(
                nusc, sample_data, inv_pose, args.datapath, 
                downsample=args.downsample
            )
            
            # 保存图像文件
            im_name = f"{i:05d}.jpg"
            cv2.imwrite(os.path.join(args.out, "images", cam, im_name), im)
            
            # 记录帧元数据
            timestamp = sample["timestamp"] / 1e6
            if start_time < 0:  # 设置初始时间戳
                start_time = timestamp
            meta_data['frames'].append({
                "rgb_path": os.path.join("./images", cam, im_name),
                "camtoworld": pose.tolist(),  # 相机到世界坐标系的变换矩阵
                "intrinsics": intrinsic.tolist(),  # 相机内参矩阵
                "width": width,
                "height": height,
                'timestamp': timestamp - start_time,  # 相对时间戳
                "dynamics": dynamics,  # 当前帧中的动态物体位姿
            })

            # 视频生成处理
            if args.video:
                im_rgb = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
                im_resized = cv2.resize(im_rgb, (400, 225))  # 统一尺寸
                cat_images.append(im_resized)

        # 拼接全景视频帧（6相机布局）
        if args.video and len(cat_images) == 6:
            panorama = cv2.vconcat([
                cv2.hconcat([cat_images[1], cat_images[0], cat_images[2]]),  # 前中右
                cv2.hconcat([cat_images[5], cat_images[3], cat_images[4]]),  # 左后右
            ])
            video_images.append(panorama)

    ###################################################################
    #                   第三阶段：结果保存                              #
    ###################################################################
    
    # 保存元数据文件
    with open(os.path.join(args.out, 'meta_data.json'), 'w') as wf:
        json.dump(meta_data, wf, indent=2)
    
    # 生成全景视频（如果启用）
    if args.video:
        media.write_video(os.path.join(args.out, 'view.mp4'), video_images, fps=12)

    print("数据处理完成！输出目录：", args.out)
