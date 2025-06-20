import numpy as np
# import transforms3d as tr
from scipy.spatial.transform import Rotation as R  # 用于旋转矩阵计算
import cv2  # OpenCV图像处理
import json
from tqdm import tqdm  # 进度条显示
import os
import argparse
import open3d as o3d  # 用于3D点云处理
from utils import *

def get_opts():
    """解析命令行参数"""
    parser = argparse.ArgumentParser()
    parser.add_argument('-b', '--base_path', type=str, required=True,
                        help='Waymo数据集的基础路径')
    parser.add_argument('-s', '--segment', type=str, required=True,
                        help='要处理的数据段ID')
    parser.add_argument('--track_seq_id', type=int, required=True,
                        help='要处理的track数据段ID')
    parser.add_argument('-o', '--outpath', type=str, required=True,
                        help='输出目录路径')
    parser.add_argument('--downsample', type=float, default=2,
                        help='图像下采样率，默认为2')
    parser.add_argument('--rimg', action="store_true", default=False,
                        help='是否生成rimg图像'),
    parser.add_argument('--lidar_depth', action="store_true", default=True,
                        help='是否根据点云生成深度图像')

    return parser.parse_args()


if __name__ == '__main__':
    # 解析命令行参数
    args = get_opts()

    # 构建数据段的完整路径
    seq_path = os.path.join(args.base_path, f"{args.segment}")
    track_seq = f"{args.segment}_{args.track_seq_id}"  # 组合segment和track_seq
    # 获取JSON文件路径 带标定信息的追踪结果 100ms一帧，每个文件夹含有150帧，即15s的数据
    json_dir = os.path.join(
        seq_path,
        f"annotations/trackOD/{track_seq}/cache/autolabel_4dod/{track_seq}/offline_showformat"
    )

    # 获取所有JSON文件（假设已按时间戳排序）
    json_files = sorted([os.path.join(json_dir, f)
                        for f in os.listdir(json_dir) if f.endswith('.json')])

    # 获取15秒时间窗口内的文件（前15秒）
    filtered_files = json_files[:50]  # 10fps，15秒共150帧

    # 预处理odom数据
    import bisect
    odom_file = os.path.join(seq_path, "odom_5.txt")
    odom_data = []
    odom_timestamps = []
    if os.path.exists(odom_file):
        with open(odom_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 7:  # 时间戳 + x,y,z,roll,pitch,yaw
                    timestamp = float(parts[0]) + 28800  # 添加8小时时区校正
                    odom_data.append({
                        'timestamp': timestamp,
                        'x': float(parts[1]),
                        'y': float(parts[2]),
                        'z': float(parts[3]),
                        'roll': float(parts[4]),
                        'pitch': float(parts[5]),
                        'yaw': float(parts[6])
                    })
                    odom_timestamps.append(timestamp)

    # 从meta.json中获取车辆位姿
    meta_file = os.path.join(seq_path, "meta.json")
    ego2world = {}
    with open(meta_file, 'r') as f:
        data = json.load(f)
    meta = data.get("meta", {})
    for i, (frame_name, frame_data) in enumerate(meta.items()):
        # 提取前 150 帧的信息
        if i >= 150:
            break
        pose_list = frame_data.get("pose", [])
        matrix4 = None
        if isinstance(pose_list, list) and pose_list:
            matrix4 = pose_list[0].get("matrix4")
            ego2world[frame_name] = matrix4

    # 创建可遍历的数据序列对象
    class JSONSequence:
        def __init__(self, files):
            self.files = files
            self.current_idx = 0

        def __iter__(self):
            return self

        def __next__(self):
            if self.current_idx >= len(self.files):
                raise StopIteration
            with open(self.files[self.current_idx]) as f:
                data = json.load(f)
            self.current_idx += 1
            return data

    # 创建序列对象
    data_sequence = JSONSequence(filtered_files)

    # 创建输出目录结构
    save_dir = args.outpath
    os.makedirs(save_dir, exist_ok=True)
    cams=AVAILABLE_CAMERAS
    # 为每个相机创建图像保存目录
    for cam in cams:
        os.makedirs(os.path.join(save_dir, "images", f"{cam}"), exist_ok=True)

    # 创建Lidar保存目录
    os.makedirs(os.path.join(save_dir, "lidar"), exist_ok=True)
    # 创建有颜色点云的保存目录
    os.makedirs(os.path.join(save_dir, "lidar_colored"), exist_ok=True)

    ##########################################################################
    #                     读取所有帧信息                                      #
    ##########################################################################
    # 初始化数据存储字典
    ego_poses = {}  # 车辆位姿
    extr = {}       # 相机外参
    c2ws = {}       # 相机到世界的变换
    intr = {}       # 相机内参
    imsize = {}     # 图像尺寸
    vehicles = {}   # 车辆信息
    dynamics = {}   # 动态物体信息
    timestamps = []  # 时间戳列表
    start_timestamp = None
    first_pcd = None
    valid_frame_idx = 0

    # 遍历所有JSON数据帧
    for frame_idx, frame_data in tqdm(enumerate(data_sequence)):
        # 计算相对时间戳（秒）
        timestamp_str = frame_data["meta"]["sensor"][0]["timestamp"]

        # 获取自车在世界中的位姿（车辆到世界的变换）
        v2w = np.eye(4)
        # 优先使用meta json中匹配到的矩阵
        if timestamp_str in ego2world:
            v2w = np.array(ego2world[timestamp_str]).reshape(4, 4)
        else:
            # 有时 meta json 中没有，该帧落盘有问题，跳过该帧
            continue
            # # 有时 meta json 中没有，则在odom中找对应时间戳的车辆位姿
            # idx = bisect.bisect_left(odom_timestamps, current_timestamp)
            # # 处理边界情况
            # if idx == 0:
            #     closest_odom = odom_data[0]
            # elif idx == len(odom_data):
            #     closest_odom = odom_data[-1]
            # else:
            #     # 比较前后两个时间戳
            #     prev_diff = current_timestamp - odom_timestamps[idx-1]
            #     next_diff = odom_timestamps[idx] - current_timestamp
            #     closest_odom = odom_data[idx -
            #                              1] if prev_diff < next_diff else odom_data[idx]
            # if current_timestamp-closest_odom['timestamp'] < 1e-6:
            #     rot = tr.euler.euler2mat(
            #         closest_odom['roll'],
            #         closest_odom['pitch'],
            #         closest_odom['yaw'])

            #     v2w[:3, :3] = rot
            #     v2w[:3, 3] = [
            #         closest_odom['x'],
            #         closest_odom['y'],
            #         closest_odom['z']
            #     ]
            # else:
            #     raise ValueError(
            #         f"Timestamp '{timestamp_str}' not found in mete_json or odom5_txt!")

        current_timestamp = parse_timestamp(timestamp_str)
        if start_timestamp is None:
            start_timestamp = current_timestamp
            first_pcd = timestamp_str
        t = current_timestamp - start_timestamp

        timestamps.append(t)

        # 处理Lidar数据
        # 读取PCD文件
        pcd_name = timestamp_str
        pcd_path = os.path.join(
            seq_path, f"Lidar/Pandar128_Compensate/{pcd_name}")
        pcd = o3d.io.read_point_cloud(pcd_path)

        # 将点云数据转换为numpy数组，只保留xyz坐标
        lidar_points = np.asarray(pcd.points)[:, :3]
        # # 筛选车辆附近可能的地面点  车辆前后 6米范围内（|X| < 6）车辆左右 3米范围内（|Y| < 3）
        # ground_mask = (np.abs(lidar_points[:, 0]) < 6) & (
        #     np.abs(lidar_points[:, 1]) < 3)
        # lidar_points = lidar_points[ground_mask]

        # 创建Open3D点云对象
        pcd = o3d.geometry.PointCloud()
        # 初始化点云颜色数组，默认为黑色
        colors = np.zeros((len(lidar_points), 3))
        # 记录每个点是否已经被赋予颜色
        color_assigned = np.zeros(len(lidar_points), dtype=bool)

        # 处理图像数据
        for sensor in frame_data.get('meta', {}).get('sensor', []):
            cam_id = sensor.get('sensor_id')
            if cams and cam_id not in cams:
                continue
            data_path = sensor.get('data_path')
            if not data_path:
                print(
                    f"Error: Failed to read image path for {cam_id} at {timestamp_str}")
                continue
            split_path = data_path.split('/')
            relative_path = split_path[3:]
            img_path = os.path.join(seq_path, *relative_path)
            img = cv2.imread(img_path)
            if img is None:
                print(
                    f"Warning: Failed to read image for {cam_id} at {img_path}")
                continue
            h, w = img.shape[:2]
            downsample_factor = args.downsample
            
            # 对称截去自车车体部分
            ego_body_pixels = 0
            if cam_id == 'CAM_FRONT_120':
                ego_body_pixels = 512  # 前视Dedistort图引擎盖高度
                # img = img[0:h, :]
                img = img[ego_body_pixels:h-ego_body_pixels, :]
                h = h-2*ego_body_pixels
            elif cam_id == 'CAM_BACK':
                ego_body_pixels = 56  # 后视Dedistort图车尾高度
                # img = img[0:h, :]
                img = img[ego_body_pixels:h-ego_body_pixels, :]
                h = h-2*ego_body_pixels                
           
            if downsample_factor > 1:
                h = int(h // downsample_factor)
                w = int(w // downsample_factor)
                img = cv2.resize(img, (w, h))
            output_dir = os.path.join(save_dir, "images", cam_id)
            os.makedirs(output_dir, exist_ok=True)
            output_path = os.path.join(
                output_dir, f"{str(valid_frame_idx).zfill(6)}.png")
            cv2.imwrite(output_path, img)
            if cam_id not in imsize:
                imsize[cam_id] = []
            imsize[cam_id].append((h, w))

            # 获取相机参数
            params = sensor['sensor_param']
            # 构建相机内参矩阵（考虑下采样和截取）
            cam_intrinsic = np.eye(4)
            cam_intrinsic[0, 0] = params['intrinsic'][0][0] / downsample_factor
            cam_intrinsic[1, 1] = params['intrinsic'][1][1] / downsample_factor
            cam_intrinsic[0, 2] = params['intrinsic'][0][2] / downsample_factor
            cam_intrinsic[1, 2] = (
                params['intrinsic'][1][2] - ego_body_pixels) / downsample_factor

            if cam_id not in intr:
                intr[cam_id] = []
            intr[cam_id].append(cam_intrinsic)

            # 从四元数(wxyz)构建旋转矩阵
            quat = params['sensor2ego_rotation']
            rot = R.from_quat(
                [quat[1], quat[2], quat[3], quat[0]]).as_matrix()

            # 构建相机到车辆的变换矩阵
            c2v = np.eye(4)
            c2v[:3, :3] = rot
            c2v[:3, 3] = params['sensor2ego_translation']

            if cam_id not in extr:
                extr[cam_id] = []
            extr[cam_id].append(c2v)

            # 暂时使用当前帧的位姿作为相机时刻的自车世界位置
            if cam_id not in ego_poses:
                ego_poses[cam_id] = []
            ego_poses[cam_id].append(v2w)

            # 获取相机参数
            c2v = extr[cam_id][valid_frame_idx]  # 相机到自车的变换
            v2c = np.linalg.inv(c2v)  # 自车到相机的变换
            K = intr[cam_id][valid_frame_idx]  # 相机内参

            if (args.lidar_depth):
                # 将点云从自车坐标系转换到相机坐标系
                points_cam = (v2c[:3, :3] @ lidar_points.T).T + v2c[:3, 3]
                # 保存转换到前视120坐标系下的点云
                if (cam_id == 'CAM_FRONT_120'):
                    points_cam_front120 = points_cam

                # 过滤掉相机后方的点(z < 0)
                front_mask = points_cam[:, 2] > 0
                points_cam = points_cam[front_mask]

                # 将点云投影到图像平面
                points_img = (K[:3, :3] @ points_cam.T).T + K[:3, 3]
                points_uv = (points_img[:, :2] /
                            points_img[:, 2][:, None]).astype(int)

                # 获取每个点的深度值
                depths = points_cam[:, 2]

                # 创建深度图目录
                depth_dir = os.path.join(save_dir, "lidar_depth", cam_id)
                os.makedirs(depth_dir, exist_ok=True)
                depth_path = os.path.join(
                    depth_dir, f"{str(valid_frame_idx).zfill(6)}.npy")
                # 创建稀疏深度图
                depth_map = np.zeros((h, w), dtype=np.float32)
                # 填充稀疏深度图
                for i, uv in enumerate(points_uv):
                    if 0 <= uv[0] < w and 0 <= uv[1] < h:
                        depth_map[uv[1], uv[0]] = depths[i]
                # 保存深度图为numpy数组
                np.save(depth_path, depth_map)

                # # 创建深度图可视化
                # depth_vis = np.zeros_like(depth_map, dtype=np.uint8)
                # mask = depth_map > 0

                # # 使用jet颜色映射进行可视化
                # if np.any(mask):
                #     # 使用彩色映射进行可视化
                #     depth_color = cv2.applyColorMap(
                #         depth_vis, cv2.COLORMAP_JET)
                #     # 将没有深度值的区域设为黑色
                #     depth_color[~mask] = [0, 0, 0]
                #     # 保存彩色深度图可视化
                #     color_vis_path = os.path.join(
                #         depth_dir, f"{str(valid_frame_idx).zfill(6)}_vis.png")
                #     cv2.imwrite(color_vis_path, depth_color)

                # 检查点是否在图像范围内
                valid_mask = (points_uv[:, 0] >= 0) & (points_uv[:, 0] < w) & \
                    (points_uv[:, 1] >= 0) & (points_uv[:, 1] < h)

                # 获取原始点云中的索引
                valid_indices = np.where(front_mask)[0][valid_mask]

                # 为未赋值的有效点采样颜色
                for idx, uv in zip(valid_indices, points_uv[valid_mask]):
                    if not color_assigned[idx]:
                        # 从图像中采样BGR颜色并转换为RGB
                        color = img[uv[1], uv[0], ::-1] / 255.0
                        colors[idx] = color
                        color_assigned[idx] = True

                if (args.rimg):
                    # 创建rimg目录并保存点云投影图像
                    rimg_dir = os.path.join(save_dir, "rimg", cam_id)
                    os.makedirs(rimg_dir, exist_ok=True)
                    rimg_path = os.path.join(
                        rimg_dir, f"{str(valid_frame_idx).zfill(6)}.png")

                    # 创建rimg图像
                    rimg = img.copy()
                    # 绘制投影点,使用基于深度的颜色
                    for i, uv in enumerate(points_uv):
                        if 0 <= uv[0] < w and 0 <= uv[1] < h:
                            # 获取基于深度的RGB颜色
                            color = get_rgb_by_distance(
                                depths[i], min_val=0, max_val=100)
                            # OpenCV使用BGR顺序,需要反转RGB
                            color_bgr = (int(color[2]), int(
                                color[1]), int(color[0]))
                            cv2.circle(rimg, tuple(uv), 1, color_bgr, -1)
                    cv2.imwrite(rimg_path, rimg)

                # 保存完整点云数据到lidar目录
                pcd.points = o3d.utility.Vector3dVector(points_cam_front120)
                pcd.colors = o3d.utility.Vector3dVector(colors)
                output_dir = os.path.join(save_dir, "lidar")
                output_path = os.path.join(
                    output_dir, f"{str(valid_frame_idx).zfill(6)}.ply")
                o3d.io.write_point_cloud(output_path, pcd)

                # 只保存有颜色的点云数据到lidar_colored目录
                valid_points = points_cam_front120[color_assigned]
                valid_colors = colors[color_assigned]
                colored_pcd = o3d.geometry.PointCloud()
                colored_pcd.points = o3d.utility.Vector3dVector(valid_points)
                colored_pcd.colors = o3d.utility.Vector3dVector(valid_colors)
                colored_output_dir = os.path.join(save_dir, "lidar_colored")
                colored_output_path = os.path.join(
                    colored_output_dir, f"{str(valid_frame_idx).zfill(6)}.ply")
                o3d.io.write_point_cloud(colored_output_path, colored_pcd)

        # 处理3D边界框标注
        for obj in frame_data['annotations']:
            # 获取对象ID和类型，如果字段不存在则抛出错误
            if 'property' not in obj or 'track_id' not in obj['property']:
                raise ValueError(
                    "Missing required field: property.instance_id")
            obj_id = str(obj['property']['track_id'])

            if 'category' not in obj:
                raise ValueError("Missing required field: category")
            type_name = obj['category']

            # 只处理车辆、行人和骑车人
            if type_name in ['MotorVehicle', 'Pedestrian', 'TwoWheels', 'Tricycle']:
                # 从PC_3D数组中获取边界框信息
                # [x, y, z, w, l, h, roll, pitch, yaw, vx, vy, vz]
                pc3d = obj['PC_3D']
                # 获取边界框中心点坐标，自车坐标系下
                x, y, z = pc3d[0:3]
                # 获取边界框尺寸（x→l, y→w, z->h）
                width, length, height = pc3d[3:6]  # w, l, h
                # 获取旋转角度
                roll, pitch, yaw = pc3d[6:9]

                # 计算Box到自车的转换矩阵b2v
                # 首先创建旋转矩阵，考虑roll、pitch、yaw
                rot_matrix = R.from_euler(
                    'zxy', [yaw, roll, pitch]).as_matrix()

                b2v = np.eye(4)
                b2v[:3, :3] = rot_matrix
                b2v[:3, 3] = [x, y, z]

                # 边界框从自车坐标系转换到世界坐标系
                b2w = v2w @ b2v @ LHW_TO_LWH

                if obj_id not in vehicles:
                    vehicles[obj_id] = {
                        "rt": [],          # 变换信息
                        "lhw": [],         # 长、高、宽
                        "timestamp": [],   # 时间戳
                        "frame": [],       # 帧索引
                    }

                vehicles[obj_id]['rt'].append(b2w)
                vehicles[obj_id]['lhw'].append(
                    np.array([length, height, width]))
                vehicles[obj_id]["timestamp"].append(t)
                vehicles[obj_id]['frame'].append(valid_frame_idx)

        valid_frame_idx += 1
    # 遍历每帧json数据结束

    # 标准化位姿 - 将所有位姿转换到前视相机的第0帧坐标系下
    # 计算每个相机在每一帧的c2w
    for cam, v2ws in ego_poses.items():
        for i, v2w in enumerate(v2ws):
            c2v = extr[cam][i]  # 相机到车辆的变换
            # 计算相机到世界的变换： 相机到车辆-> 车辆到世界
            c2w = v2w @ c2v
            if cam not in c2ws:
                c2ws[cam] = []
            c2ws[cam].append(c2w)

    # 计算相机CAM_FRONT_120第0帧的逆变换矩阵（作为参考坐标系）
    inv_pose = np.linalg.inv(c2ws['CAM_FRONT_120'][0])

    # 将所有相机位姿转换到参考坐标系
    for cam, poses in c2ws.items():
        poses = np.stack(poses)
        # 应用逆变换： inv_pose @ poses
        poses = np.einsum('njk,ij->nik', poses, inv_pose)
        c2ws[cam] = poses
    # ！此后，从c2ws获取的c2w就是从相机到参考坐标系的变换矩阵

    # 筛选动态物体（移动超过1米的物体）
    dynamic_id = 0
    for objid, infos in vehicles.items():
        # 将物体位置信息转换为数组
        infos['rt'] = np.stack(infos['rt'])
        # 提取位置信息（变换矩阵的平移部分）
        trans = infos['rt'][:, :3, 3]  # 提取变换矩阵中的平移向量
        # 将位置从世界坐标转换到参考坐标系
        trans = np.einsum('nj,ij->ni', trans,
                          inv_pose[:3, :3]) + inv_pose[:3, 3]
        # 计算物体在序列中的最大移动距离
        movement = np.max(np.max(trans, axis=0) - np.min(trans, axis=0))
        # 如果移动超过1米，认为是动态物体
        if movement > 1:
            dynamics[dynamic_id] = infos
            dynamic_id += 1

    # 后处理动态物体信息
    verts = {}  # 存储物体顶点信息
    rts = {}    # 存储物体变换矩阵
    for dynamic_id, infos in dynamics.items():
        # 获取物体尺寸（长、高、宽）
        lhw = np.array(infos['lhw'][0])
        # 计算边界框顶点
        points = get_vertices(lhw)
        # 获取物体位姿变换矩阵(b2w)
        b2ws = infos['rt']

        seq_visible = False  # 物体在整个序列中是否可见
        # 遍历物体出现的所有帧
        for idx, fid in enumerate(infos['frame']):
            # 物体到参考系的变换矩阵
            rt = inv_pose @ b2ws[idx]

            # 将边界框顶点转换到世界坐标系（参考系）
            points_w = (rt[:3, :3] @ points.T).T + rt[:3, 3]

            frame_visible = False  # 物体在当前帧是否可见
            # 检查物体在每个相机中是否可见
            for cam in cams:
                # 相机到参考系
                c2w = c2ws[cam][fid]
                w2c = np.linalg.inv(c2w)  # 世界到相机的变换
                K = intr[cam][fid]        # 相机内参
                h, w = imsize[cam][fid]   # 图像尺寸

                # 将边界框顶点转换到相机坐标系
                points_cam = (w2c[:3, :3] @ points_w.T).T + w2c[:3, 3]
                # 将顶点投影到图像平面
                points_screen = (K[:3, :3] @ points_cam.T).T + K[:3, 3]
                # 计算顶点的像素坐标
                points_uv = (
                    points_screen[:, :2] / points_screen[:, 2][:, None]).astype(int)

                # 检查顶点是否在图像范围内
                valid_mask = (points_screen[:, 2] > 0) & (points_uv[:, 0] >= 0) & (
                    points_uv[:, 1] >= 0) & (points_uv[:, 0] < w) & (points_uv[:, 1] < h)
                if np.sum(valid_mask) > 0:
                    # 如果至少有一个顶点在图像中可见
                    frame_visible = True
                    seq_visible = True
                    break

            # 如果物体在当前帧可见，保存其变换矩阵
            if frame_visible:
                if fid not in rts:
                    rts[fid] = {}
                rts[fid][dynamic_id] = rt.tolist()

        # 如果物体在整个序列中可见，保存其顶点信息
        if seq_visible:
            verts[dynamic_id] = points.tolist()

    # 生成元数据JSON文件
    meta_data = {
        "camera_model": "OPENCV",  # 使用OpenCV相机模型
        "frames": [],              # 帧信息列表
        "verts": verts,            # 动态物体顶点信息
        "inv_pose": inv_pose.tolist()  # 参考坐标系的逆变换矩阵
    }

    # 为每一帧和每个相机生成元数据
    for i in range(len(intr['CAM_FRONT_120'])):  # 使用相机的帧数作为基准
        for cam in cams:
            # 获取相机参数
            intrinsic = intr[cam][i]    # 内参
            camtoworld = c2ws[cam][i]   # 相机到世界的变换
            h, w = imsize[cam][i]       # 图像尺寸

            # 构建帧信息
            info = {
                'rgb_path': f'./images/{cam}/{str(i).zfill(6)}.png',  # 图像路径
                'camtoworld': camtoworld.tolist(),  # 相机位姿
                'intrinsics': intrinsic.tolist(),   # 相机内参
                'width': w,                         # 图像宽度
                'height': h,                        # 图像高度
                'timestamp': timestamps[i],         # 时间戳
                "dynamics": rts.get(i, {})          # 当前帧中的动态物体信息
            }
            meta_data['frames'].append(info)

    # 保存元数据到JSON文件
    with open(os.path.join(save_dir, 'meta_data.json'), 'w') as wf:
        json.dump(meta_data, wf, indent=2)

    ##########################################################################
    #                        读取第一帧信息                                    #
    #           激光雷达数据仅用于提取相机高度                                   #
    #           计算相机相对于地面的高度（沿 z 轴）                              #
    ##########################################################################

    # 读取PCD文件
    pcd_path = os.path.join(
        seq_path, f"Lidar/Pandar128_Compensate/{first_pcd}")
    pcd = o3d.io.read_point_cloud(pcd_path)
    # 将点云数据转换为numpy数组，只保留xyz坐标
    lidar_points = np.asarray(pcd.points)[:, :3]
    # 筛选车辆附近可能的地面点  车辆前后 6米范围内（|X| < 6）车辆左右 3米范围内（|Y| < 3）
    ground_mask = (np.abs(lidar_points[:, 0]) < 6) & (
        np.abs(lidar_points[:, 1]) < 3)
    lidar_points = lidar_points[ground_mask]

    # 创建Open3D点云对象
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(lidar_points)
    # 保存地面点云数据
    o3d.io.write_point_cloud(os.path.join(
        args.outpath, 'ground_lidar.ply'), pcd)

    # 使用RANSAC算法分割地面平面
    # 平面方程: ax + by + cz + d = 0
    plane_model, inliers = pcd.segment_plane(distance_threshold=0.01,
                                             ransac_n=3,
                                             num_iterations=1000)
    a, b, c, d = plane_model

    # 获取前置相机的位置
    front_cam_t = extr['CAM_FRONT_120'][0][:3, 3]
    # 计算相机到地面的高度
    # 使用平面方程计算点到平面的距离
    height = -(a*front_cam_t[0] + b*front_cam_t[1] + d) / c
    front_cam_info = {
        "height": front_cam_t[2] - height,  # 相机高度
        "rect_mat": None,  # 矫正矩阵（未使用）
    }
    # 保存前置相机信息
    with open(os.path.join(args.outpath, 'front_info.json'), 'w') as f:
        json.dump(front_cam_info, f, indent=2)

    # save camera relative pose for rigid bundle adjustment
    cam_rigid = dict()
    cam_rigid["ref_camera_id"] = 1
    rigid_cam_list = []
    ref_extrinsic =  extr['CAM_FRONT_120'][0]
    for iid, cam_name in enumerate(AVAILABLE_CAMERAS):
        rigid_cam = dict()
        rigid_cam["camera_id"] = iid+1

        cur_extrinsic =  extr[cam_name][0]
        rel_extrinsic = np.linalg.inv(cur_extrinsic) @ ref_extrinsic
        r = R.from_matrix(rel_extrinsic[:3, :3])
        qvec = r.as_quat()
        rigid_cam["image_prefix"] = f'{cam_name}/'        
        rigid_cam['cam_from_rig_rotation'] = [qvec[3], qvec[0], qvec[1], qvec[2]]
        rigid_cam['cam_from_rig_translation'] = [rel_extrinsic[0, 3], rel_extrinsic[1, 3], rel_extrinsic[2, 3]]
        rigid_cam_list.append(rigid_cam)

    cam_rigid["cameras"] = rigid_cam_list
    with open(os.path.join(args.outpath, "cam_rigid_config.json"), "w") as f:
        json.dump([cam_rigid], f, indent=4)           
