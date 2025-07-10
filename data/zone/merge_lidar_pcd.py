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
import CSF
import pickle

def save_point_cloud(points, colors, filename, verbose=True):
    """
    保存点云数据到PLY文件
    
    参数:
        points: 点云坐标数组 (N,3)
        colors: 点云颜色数组 (N,3) 或 None
        filename: 保存的文件路径
        verbose: 是否打印保存信息
    """
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    
    if colors is not None:
        # 确保颜色值在[0,1]范围内
        if colors.max() > 1.0:
            colors = colors.astype(np.float32) / 255.0
        pcd.colors = o3d.utility.Vector3dVector(colors)
    
    o3d.io.write_point_cloud(filename, pcd)
    
    if verbose:
        print(f"保存点云到 {filename} (点数: {len(points)})")

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


def segment_ground_ransac(points, colors, max_distance=0.4, max_iterations=1000, return_mask=False):
    """
    在相机坐标系中分割地面点云（地面是y轴最大的平面）
    参数:
        points: 点云坐标数组 (N,3)
        colors: 点云颜色数组 (N,3)
        max_distance: 点到平面的最大距离阈值
        max_iterations: RANSAC最大迭代次数
        return_mask: 是否返回地面点掩码
    返回:
        如果return_mask为False:
            (非地面点云, 对应颜色, 地面点云, 对应颜色)
        如果return_mask为True:
            (非地面点云, 对应颜色, 地面点云, 对应颜色, 地面点掩码)
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

    # 获取地面点掩码（inliers）和非地面点掩码
    ground_mask = ransac.inlier_mask_
    non_ground_mask = ~ground_mask

    # 提取地面点云和非地面点云
    non_ground_points = points[non_ground_mask]
    non_ground_colors = colors[non_ground_mask]
    ground_points = points[ground_mask]
    ground_colors = colors[ground_mask]

    if return_mask:
        return non_ground_points, non_ground_colors, ground_points, ground_colors, ground_mask
    else:
        return non_ground_points, non_ground_colors, ground_points, ground_colors


def segment_ground_csf(points, colors, cloth_resolution=0.5, rigidness=2,
                       class_threshold=0.5, iterations=500, **kwargs):
    """
    使用CSF算法在相机坐标系中分割地面点云（y轴朝下）
    
    参数:
        points: 点云坐标数组 (N,3) - 相机坐标系(y轴向下)
        colors: 点云颜色数组 (N,3)
        cloth_resolution: 布料网格分辨率(米)
        rigidness: 布料刚性(1-3, 1最软)
        class_threshold: 点分类阈值
        iterations: 布料模拟迭代次数
        
    返回:
         (非地面点云, 对应颜色, 地面点云, 对应颜色)
    """
    # 1. 坐标系转换：相机坐标系(y向下) → CSF标准坐标系(z向上)
    # 转换关系: [x, y, z] → [x, -z, -y]
    # 解释:
    #   - 将y轴(向下)映射为z轴(向上)并取反
    #   - 将z轴(前向)映射为y轴(水平)
    csf_points = np.zeros_like(points)
    csf_points[:, 0] = points[:, 0]   # x保持不变
    csf_points[:, 1] = -points[:, 2]  # z取反作为y
    csf_points[:, 2] = -points[:, 1]  # y取反作为z(向上)

    # 2. 配置CSF参数
    csf = CSF.CSF()
    csf.params.bSloopSmooth = True     # 启用坡度平滑
    csf.params.cloth_resolution = cloth_resolution
    csf.params.rigidness = rigidness
    csf.params.class_threshold = class_threshold
    csf.params.iterations = iterations

    # 3. 设置点云并执行过滤
    csf.setPointCloud(csf_points)
    ground = CSF.VecInt()  # a list to indicate the index of ground points after calculation
    non_ground = CSF.VecInt() # a list to indicate the index of non-ground points after calculation
    csf.do_filtering(ground, non_ground) # do actual filtering.

    # 4. 提取非地面点(在原坐标系中)
    non_ground_points = points[non_ground]
    non_ground_colors = colors[non_ground]

    ground_points = points[ground]
    ground_colors = colors[ground]

    return non_ground_points, non_ground_colors,ground_points,ground_colors

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
    ground_points,ground_colors=[],[]
    non_ground_points,non_ground_colors=[],[]
    # 初始化全局存储列表
    all_ground_points = []
    all_ground_colors = []
    all_non_ground_points = []
    all_non_ground_colors = []
    
    sample_per_frame = args.total // len(meta_data["frames"]) * 6
    front_cam_poses = []
    last_frame_idx=None
    frame_points, frame_colors=[],[]
    
    # 初始化当前帧的语义地面点存储
    current_frame_data = {
        'ground_mask': None,  # 当前帧语义地面点mask
        'local_points_w': None,  # 当前帧世界坐标系下的原始点云
        'local_colors': None  # 当前帧原始颜色
    }

    for frame in tqdm(meta_data["frames"]):
        rgb_path = frame["rgb_path"]
        # 获取帧索引，从rgb_path中提取
        frame_idx = int(rgb_path.split("/")[-1].split(".")[0])
        frame_cam = frame["rgb_path"].split("/")[-2]

        K = np.array(frame["intrinsics"])
        c2w = np.array(frame["camtoworld"])
        w2c=np.linalg.inv(c2w)
        H, W = frame["height"], frame["width"]

        # 新的一帧
        if frame_idx != last_frame_idx and frame_cam == 'CAM_FRONT_120':
            last_frame_idx = frame_idx

            front_cam_poses.append(c2w)

            # 构建对应的ply文件路径
            ply_path = os.path.join(args.out, "lidar_colored",
                                    f"{str(frame_idx).zfill(6)}.ply")
            # 检查文件是否存在
            if not os.path.exists(ply_path):
                print(f"Warning: PLY file not found: {ply_path}")
                continue
            # 获取当前帧CAM_FRONT_120 相机坐标系下的点云
            pcd = o3d.io.read_point_cloud(ply_path)
            #  如果点云数量不足，跳过该帧
            if np.asarray(pcd.points).shape[0] < sample_per_frame:
                print(
                    f"Warning: Not enough points in frame {frame_idx}: {local_points.shape[0]} < {sample_per_frame}")
                continue

            # # 降采样
            # 体素滤波
            down_pcd = pcd.voxel_down_sample(voxel_size=0.15)
            # 去除离群点
            # down_pcd, _ = down_pcd.remove_radius_outlier(nb_points=10, radius=0.5)

            # 获取下采样后的点云和颜色
            local_points = np.asarray(down_pcd.points)
            local_colors = np.asarray(down_pcd.colors)

            # 1.去除打到ego的点
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

                # 初始化当前帧的语义地面点mask
                current_frame_data['ground_mask'] = np.zeros(len(local_points_w), dtype=bool)
                current_frame_data['local_points_w'] = local_points_w
                current_frame_data['local_colors'] = local_colors        

        # 3.分割地面点
        # 加载语义分割图
        smts_path = os.path.join(
            args.out,
            rgb_path.replace("images", "semantics")
            .replace("./", "")
            .replace(".jpg", ".npy")
            .replace(".png", ".npy"),
        )
    
        # 处理当前相机的语义地面点
        if os.path.exists(smts_path):
            smts = np.load(smts_path)

            # 转换到当前相机坐标系
            local_points=(w2c[:3, :3] @ local_points_w.T).T+w2c[:3, 3]

            # 只处理相机前方的点
            front_mask = local_points[:, 2] > 0
            points_cam = local_points[front_mask]

            # 投影到图像平面
            points_img = (K[:3, :3] @ points_cam.T).T + K[:3, 3]
            points_img = points_img[:, :2] / points_img[:, 2:]
        
            # 创建有效点mask (在图像边界内)
            valid_mask = np.zeros(len(local_points), dtype=bool)  # 初始化全False
            img_valid_mask = (points_img[:, 0] >= 0) & (points_img[:, 0] < W) & \
                            (points_img[:, 1] >= 0) & (points_img[:, 1] < H)
            valid_mask[front_mask] = img_valid_mask
        
            # 初始化语义标签数组
            point_semantics = np.zeros(len(local_points), dtype=np.int32) - 1  # -1表示无效
        
            # 为有效点获取语义标签
            if valid_mask.any():
                img_coords = points_img[img_valid_mask].astype(int)
                point_semantics[valid_mask] = smts[img_coords[:, 1], img_coords[:, 0]]
        
            # 当前相机的语义地面mask (标签为0或1)
            current_cam_mask = (point_semantics == 0) | (point_semantics == 1)
        
            current_cam_semantic_ground_points = local_points[current_cam_mask]
            current_cam_semantic_ground_colors = local_colors[current_cam_mask]

            # os.makedirs(os.path.join(args.out, "lidar_seg"), exist_ok=True)
            # save_point_cloud(
            #     current_cam_semantic_ground_points,
            #     current_cam_semantic_ground_colors,
            #     filename=os.path.join(args.out, f"lidar_seg/{str(frame_idx).zfill(6)}_{str(frame_cam)}_sem_ground.ply"),
            #     verbose=False
            # )  

            current_cam_semantic_non_ground_points = local_points[~current_cam_mask]
            current_cam_semantic_non_ground_colors = local_colors[~current_cam_mask]

            # os.makedirs(os.path.join(args.out, "lidar_seg"), exist_ok=True)
            # save_point_cloud(
            #     current_cam_semantic_non_ground_points,
            #     current_cam_semantic_non_ground_colors,
            #     filename=os.path.join(args.out, f"lidar_seg/{str(frame_idx).zfill(6)}_{str(frame_cam)}_sem_non_ground.ply"),
            #     verbose=False
            # ) 

            # 累积到当前帧的mask中
            current_frame_data['ground_mask'] = current_frame_data['ground_mask'] | current_cam_mask

        else:
            print(f"Warning: Semantic segmentation not found: {smts_path}")

        # 处理到了一帧里的最后一个摄像头 
        if frame_cam == 'CAM_BACK_RIGHT':
            # 获取当前帧累积的语义地面点
            frame_ground_mask = current_frame_data['ground_mask']
            frame_points = current_frame_data['local_points_w']
            frame_colors = current_frame_data['local_colors']
        
            semantic_ground_points = frame_points[frame_ground_mask]
            semantic_ground_colors = frame_colors[frame_ground_mask]

            # os.makedirs(os.path.join(args.out, "lidar_seg"), exist_ok=True)
            # save_point_cloud(
            #     semantic_ground_points,
            #     semantic_ground_colors,
            #     filename=os.path.join(args.out, f"lidar_seg/{str(frame_idx).zfill(6)}_sem_ground.ply"),
            #     verbose=False
            # )        

            # 使用RANSAC从语义地面点中过滤误识别点
            if len(semantic_ground_points) > 0:
                # 获取RANSAC过滤结果和掩码
                _, _, filtered_ground_points, filtered_ground_colors, ransac_mask = segment_ground_ransac(
                    semantic_ground_points, semantic_ground_colors, return_mask=True)
            
                # 创建最终地面点mask（原始点云索引）
                final_ground_mask = np.zeros(len(frame_points), dtype=bool)
                final_ground_mask[frame_ground_mask] = ransac_mask
            
                # 保存过滤后的地面点
                try:
                    os.makedirs(os.path.join(args.out, "lidar_seg"), exist_ok=True)
                    save_point_cloud(
                        filtered_ground_points,
                        filtered_ground_colors,
                        filename=os.path.join(args.out, f"lidar_seg/{str(frame_idx).zfill(6)}_sem_ransac_ground.ply"),
                        verbose=False
                    )
                
                    # 获取语义地面点中被RANSAC过滤掉的点
                    rejected_from_semantic = semantic_ground_points[~ransac_mask]
                    rejected_colors = semantic_ground_colors[~ransac_mask]
                
                    # 非地面点为原始点云中非语义地面点 + 语义地面点中被RANSAC过滤掉的点
                    non_ground_mask = ~frame_ground_mask
                    non_ground_points = np.concatenate([
                        frame_points[non_ground_mask],
                        rejected_from_semantic
                    ])
                    non_ground_colors = np.concatenate([
                        frame_colors[non_ground_mask],
                        rejected_colors
                    ])
                
                    # 保存基于语义和RANSAC过滤后的非地面点
                    save_point_cloud(
                        non_ground_points,
                        non_ground_colors,
                        filename=os.path.join(args.out, f"lidar_seg/{str(frame_idx).zfill(6)}_sem_ransac_non_ground.ply"),
                        verbose=False
                    )
                except Exception as e:
                    print(f"Warning: 保存过滤结果失败: {str(e)}")
            
                # 清理当前帧的缓存数据
                current_frame_data['ground_mask'] = None
                current_frame_data['local_points_w'] = None
                current_frame_data['local_colors'] = None

            # 根据最终mask提取地面和非地面点
            frame_ground_points = frame_points[final_ground_mask]
            frame_ground_colors = frame_colors[final_ground_mask]

            frame_non_ground_points = frame_points[~final_ground_mask]
            frame_non_ground_colors = frame_colors[~final_ground_mask]

            # 将当前帧的结果追加到全局列表中
            if len(frame_ground_points) > 0:
                all_ground_points.append(frame_ground_points)
                all_ground_colors.append(frame_ground_colors)
            if len(frame_non_ground_points) > 0:
                all_non_ground_points.append(frame_non_ground_points)
                all_non_ground_colors.append(frame_non_ground_colors)
    
    # 循环结束后，合并所有帧的结果
    if len(all_ground_points) > 0:
        ground_points = np.concatenate(all_ground_points)
        ground_colors = np.concatenate(all_ground_colors)
    else:
        ground_points = np.zeros((0, 3))
        ground_colors = np.zeros((0, 3))

    if len(all_non_ground_points) > 0:
        non_ground_points = np.concatenate(all_non_ground_points)
        non_ground_colors = np.concatenate(all_non_ground_colors)
    else:
        non_ground_points = np.zeros((0, 3))
        non_ground_colors = np.zeros((0, 3))

    # 保存非地面点云 - 添加离群点检测过滤残留地面点
    if len(non_ground_points) > 0:
        # 转换为Open3D点云
        non_ground_pcd = o3d.geometry.PointCloud()
        non_ground_pcd.points = o3d.utility.Vector3dVector(non_ground_points)
        non_ground_pcd.colors = o3d.utility.Vector3dVector(non_ground_colors)
        
        # 统计离群点检测 - 过滤孤立点(可能是残留地面点)
        cl, ind = non_ground_pcd.remove_statistical_outlier(
            nb_neighbors=20,  # 邻域点数
            std_ratio=2.0    # 标准差倍数
        )
        
        # 获取过滤后的点云
        filtered_non_ground = non_ground_pcd.select_by_index(ind)
        filtered_points = np.asarray(filtered_non_ground.points)
        filtered_colors = np.asarray(filtered_non_ground.colors)
        
        # 保存过滤后的非地面点云
        save_point_cloud(
            filtered_points,
            filtered_colors,
            filename=os.path.join(args.out, "points3d_lidar_wo_ground.ply")
        )
    else:
        print("Warning: 没有非地面点云数据可保存")

    # 保存地面点云
    if len(ground_points) > 0:
        save_point_cloud(
            ground_points,
            ground_colors,
            filename=os.path.join(args.out, "points3d_lidar_ground.ply")
        )
    else:
        print("Warning: 没有地面点云数据可保存")


    ##########################################################################
    #                    Multi-Plane Ground Model                       #
    ##########################################################################

    # Read front cam poses
    with open(os.path.join(args.out, "front_info.json"), "r") as f:
        front_info = json.load(f)

    front_cam_height = front_info["height"]
    front_rect_mat = front_info["rect_mat"]
    front_cam_poses = np.stack(front_cam_poses)
    # front_cam_poses[:, :3, :3] = np.einsum('ij, njk -> nik', front_rect_mat, front_cam_poses[:, :3, :3])

    # Init ground point cloud
    points_cam_dist = np.sqrt(
        np.sum(
            (ground_points[:, np.newaxis, :] -
             front_cam_poses[:-1, :3, 3][np.newaxis, :, :])
            ** 2,
            axis=-1,
        )
    )

    # nearest cam
    nearest_cam_idx = np.argmin(points_cam_dist, axis=1)
    nearest_c2w = front_cam_poses[nearest_cam_idx]  # (N, 4, 4)
    nearest_w2c = np.linalg.inv(front_cam_poses)[nearest_cam_idx]  # (N, 4, 4)
    points_local = (
        np.einsum("nij,nj->ni", nearest_w2c[:, :3, :3], ground_points)
        + nearest_w2c[:, :3, 3]
    )  # (N, 3)
    points_local[:, 1] = front_cam_height
    ground_points = (
        np.einsum("nij,nj->ni", nearest_c2w[:, :3, :3], points_local)
        + nearest_c2w[:, :3, 3]
    )  # (N, 3)

    # 保存经多地面模型调整后的地面点云
    save_point_cloud(
        ground_points,
        ground_colors,
        filename=os.path.join(args.out, "ground_points3d.ply")
    )

    # Get high level command
    forecast = 20
    threshold = 2.5
    high_level_commands = []
    for i, cam_pose in enumerate(front_cam_poses):
        if i + forecast < front_cam_poses.shape[0]:
            forecast_campose = front_cam_poses[i + forecast]
        else:
            forecast_campose = front_cam_poses[-1]
        inv_cam_pose = np.linalg.inv(cam_pose)
        forecast_in_curr = inv_cam_pose @ forecast_campose
        if forecast_in_curr[0, 3] > threshold:
            high_level_commands.append(0)  # right
        elif forecast_in_curr[0, 3] < -threshold:
            high_level_commands.append(1)  # left
        else:
            high_level_commands.append(2)  # forward

    print(high_level_commands)
    with open(os.path.join(args.out, "ground_param.pkl"), "wb") as f:
        pickle.dump((front_cam_poses, front_cam_height,
                    high_level_commands), f)