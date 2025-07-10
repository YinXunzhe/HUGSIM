import open3d as o3d
import numpy as np
import json
import os
import time
import argparse
from scipy.spatial.transform import Rotation as SCR  # 用于旋转矩阵计算
from datetime import datetime  # 用于时间戳解析
from scipy.spatial import cKDTree  # 用于空间查询优化

# 解析命令行参数
parser = argparse.ArgumentParser(description='点云可视化与过滤工具')
parser.add_argument('--pcd', type=str, help='点云文件路径')
parser.add_argument('--json', type=str, help='标注文件路径')
parser.add_argument('--method', type=str, choices=['vectorized', 'kdtree'], default='kdtree',
                    help='点云过滤优化方法: vectorized(向量化) 或 kdtree(KD树)')
parser.add_argument('--margin', type=float, default=1.5, 
                    help='KD树方法的边界框扩展系数 (默认: 1.5)')
args = parser.parse_args()

LHW_TO_LWH = np.array(
    [
        [1, 0, 0, 0],
        [0, 0, 1, 0],
        [0, -1, 0, 0],
        [0, 0, 0, 1.0],
    ]
)


def get_vertices(dim, bottom_center=np.array([0.0, 0.0, 0.0])):
    '''
    dim: length, height, width
    bottom_center: center of bottom face of 3D bounding box

    return: vertices of 3D bounding box (8*3)
    '''
    vertices = bottom_center[None, :].repeat(8, axis=0)
    vertices[:4, 0] = vertices[:4, 0] + dim[0] / 2
    vertices[4:, 0] = vertices[4:, 0] - dim[0] / 2
    vertices[[0, 1, 4, 5], 1] = vertices[[0, 1, 4, 5], 1] + dim[1]/2
    vertices[[2, 3, 6, 7], 1] = vertices[[2, 3, 6, 7], 1] - dim[1]/2
    vertices[[0, 2, 5, 7], 2] = vertices[[0, 2, 5, 7], 2] + dim[2] / 2
    vertices[[1, 3, 4, 6], 2] = vertices[[1, 3, 4, 6], 2] - dim[2] / 2

    return vertices


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
    in_x = np.abs(points_local[:, 0]) <= half_size[0]
    in_y = np.abs(points_local[:, 1]) <= half_size[1]
    in_z = np.abs(points_local[:, 2]) <= half_size[2]
    
    # 点在边界框内需要同时满足三个维度都在范围内
    return in_x & in_y & in_z


def filter_points_kdtree(points, vehicles, margin_factor=1.5):
    """
    使用KD树优化的点云过滤方法
    
    Args:
        points: 点云坐标数组 (N, 3)
        vehicles: 包含边界框信息的字典
        margin_factor: 边界框扩展系数，用于初步筛选
        
    Returns:
        numpy.ndarray: 布尔数组，表示每个点是否应该保留（不在任何边界框内）
    """
    points_to_keep = np.ones(len(points), dtype=bool)
    
    # 构建KD树
    tree = cKDTree(points)
    
    for objid, info in vehicles.items():
        # 获取物体尺寸和位置信息
        lhw = np.array(info['lhw'][0])
        b2v = info['rt'][0]
        box_center = b2v[:3, 3]
        box_rotation = b2v[:3, :3]
        
        # 计算边界框的AABB (Axis-Aligned Bounding Box)，并扩大一定比例
        # 这样可以快速排除大部分不在边界框内的点
        half_size = lhw / 2 * margin_factor
        
        # 计算AABB的8个顶点
        corners = []
        for i in range(8):
            corner = box_center.copy()
            corner += box_rotation @ np.array([
                half_size[0] * (1 if i & 1 else -1),
                half_size[1] * (1 if i & 2 else -1),
                half_size[2] * (1 if i & 4 else -1)
            ])
            corners.append(corner)
        corners = np.array(corners)
        
        # 计算AABB的最小和最大坐标
        min_bound = np.min(corners, axis=0)
        max_bound = np.max(corners, axis=0)
        
        # 使用KD树查询在AABB内的点的索引
        indices = tree.query_ball_point(box_center, np.max(np.linalg.norm(corners - box_center, axis=1)))
        
        if not indices:
            continue
            
        # 对候选点进行精确检查
        candidate_points = points[indices]
        in_box = points_in_box_vectorized(candidate_points, box_center, lhw, box_rotation)
        
        # 更新需要保留的点
        points_to_keep[indices] = points_to_keep[indices] & (~in_box)
    
    return points_to_keep


# 读取PCD文件
pcd_path = "datasets/zone/3D_data_LSJWK4095NS119733/20250317_161633_1/Lidar/Pandar128_Compensate/n000002_2025-03-17-16-21-46-650108_Pandar128.pcd"  # 替换为你的PCD文件路径
json_file = "datasets/zone/3D_data_LSJWK4095NS119733/20250317_161633_1/annotations/trackOD/20250317_161633_1_1/cache/autolabel_4dod/20250317_161633_1_1/offline_showformat/n000002_2025-03-17-16-21-46-650108_Pandar128.pcd.json"

# ===================== 安全读取 =====================


def read_json_safe(file_path):
    """
    带异常处理的读取方式
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"错误：文件 {file_path} 不存在")
        return None
    except json.JSONDecodeError as e:
        print(f"JSON 解析错误：{str(e)}")
        return None
    except Exception as e:
        print(f"未知错误：{str(e)}")
        return None


frame_data = read_json_safe(json_file)

vehicles = {}   # 车辆信息

# 处理3D边界框标注
for obj in frame_data['annotations']:
    # 获取对象ID和类型，如果字段不存在则抛出错误
    if 'property' not in obj or 'track_id' not in obj['property']:
        raise ValueError("Missing required field: property.instance_id")
    obj_id = str(obj['property']['track_id'])

    if 'category' not in obj:
        raise ValueError("Missing required field: category")
    type_name = obj['category']

    # 只处理车辆、行人和骑车人
    if type_name in ['MotorVehicle', 'Pedestrian', 'Cycle', 'MotorCycle']:
        # 从PC_3D数组中获取边界框信息
        # [x, y, z, w, l, h, roll, pitch, yaw, vx, vy, vz]
        pc3d = obj['PC_3D']
        # 获取边界框中心点坐标，自车坐标系下
        x, y, z = pc3d[0:3]
        # 获取边界框尺寸
        width, length, height = pc3d[3:6]  # w, l, h
        # 获取旋转角度
        roll, pitch, yaw = pc3d[6:9]

        # 计算Box到自车的转换矩阵b2v
        # 首先创建旋转矩阵，考虑roll、pitch、yaw
        rot_matrix = SCR.from_euler('zxy', [yaw, roll, pitch]).as_matrix()

        b2v = np.eye(4)
        b2v[:3, :3] = rot_matrix
        b2v[:3, 3] = [x, y, z]

        # # 边界框从自车坐标系转换到世界坐标系
        # b2w = v2w @ b2v

        if obj_id not in vehicles:
            vehicles[obj_id] = {
                "rt": [],          # 变换信息
                "lhw": [],         # 长、高、宽
                "timestamp": [],   # 时间戳
                "frame": [],       # 帧索引
            }

        vehicles[obj_id]['rt'].append(b2v@LHW_TO_LWH)
        vehicles[obj_id]['lhw'].append(np.array([length, height, width]))
        # vehicles[obj_id]["timestamp"].append(t)
        # vehicles[obj_id]['frame'].append(frame_idx)

pcd = o3d.io.read_point_cloud(pcd_path)

# 检查是否成功读取点云
if not pcd.points:
    print("错误：未能读取点云文件或文件为空！")
    exit()

# 打印点云信息
print("点云信息：")
print(pcd)
print("点云点数：", len(pcd.points))


# 创建可视化窗口
vis = o3d.visualization.Visualizer()
vis.create_window()

# 添加点云到窗口
vis.add_geometry(pcd)

# 遍历每个标注生成3D边界框
for objid, info in vehicles.items():

    # 获取物体尺寸（长、高、宽）
    lhw = np.array(info['lhw'][0])
    # 计算边界框顶点
    points = get_vertices(lhw)
    # 获取物体位姿变换矩阵(b2v)
    b2v = info['rt'][0]

    # 平移到中心点坐标
    points_v = (b2v[:3, :3] @ points.T).T+b2v[:3, 3]

    connections = [[0, 1], [0, 2], [1, 3], [2, 3], [4, 5], [4, 6], [5, 7], [6, 7],
                   [1, 4], [0, 5], [3, 6], [2, 7]]

    colors = [[1, 0, 0] for _ in range(len(connections))]  # 红色边框

    # 创建线框模型
    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(points_v)
    line_set.lines = o3d.utility.Vector2iVector(connections)
    line_set.colors = o3d.utility.Vector3dVector(colors)

    # 将边界框添加到可视化窗口
    vis.add_geometry(line_set)

render_opt = vis.get_render_option()
render_opt.point_size = 2.0
# 创建一个新的点云对象，用于存储删除物体后的点云
filtered_pcd = o3d.geometry.PointCloud()
filtered_pcd.points = o3d.utility.Vector3dVector([])
filtered_pcd.colors = o3d.utility.Vector3dVector([])

# 获取点云数据
points = np.asarray(pcd.points)
colors = np.asarray(pcd.colors)

# 检查是否有颜色信息，如果没有则创建默认颜色（白色）
has_colors = len(colors) > 0
if not has_colors:
    colors = np.ones((len(points), 3), dtype=np.float64)  # 默认白色

# 性能优化选项
USE_KDTREE = False  # 设置为True使用KD树优化，False使用纯向量化方法

# 标记需要保留的点
points_to_keep = np.ones(len(points), dtype=bool)

print(f"原始点云点数: {len(points)}")
print(f"边界框数量: {len(vehicles)}")
print("开始过滤点云...")

start_time = time.time()

if USE_KDTREE:
    print("使用KD树优化方法...")
    points_to_keep = filter_points_kdtree(points, vehicles)
else:
    print("使用向量化优化方法...")
    # 向量化处理：对每个边界框，一次性检查所有点
    for objid, info in vehicles.items():
        # 获取物体尺寸（长、高、宽）
        lhw = np.array(info['lhw'][0])
        # 获取物体位姿变换矩阵(b2v)
        b2v = info['rt'][0]
        
        # 获取边界框中心点和旋转矩阵
        box_center = b2v[:3, 3]
        box_rotation = b2v[:3, :3]
        
        # 向量化检查所有点是否在当前边界框内
        in_box = points_in_box_vectorized(points, box_center, lhw, box_rotation)
        
        # 更新需要保留的点（排除在边界框内的点）
        points_to_keep = points_to_keep & (~in_box)

end_time = time.time()
processing_time = end_time - start_time

# 打印详细的性能统计
print("\n性能统计:")
print(f"处理总时间: {processing_time:.4f} 秒")
print(f"每秒处理点数: {len(points) / processing_time:.2f}")
print(f"每个点平均处理时间: {(processing_time / len(points) * 1e6):.2f} 微秒")

# 创建过滤后的点云
filtered_points = points[points_to_keep]
filtered_colors = colors[points_to_keep] if has_colors else np.ones((len(filtered_points), 3), dtype=np.float64)

filtered_pcd.points = o3d.utility.Vector3dVector(filtered_points)
if has_colors or len(filtered_points) > 0:  # 只在有颜色信息或有点时设置颜色
    filtered_pcd.colors = o3d.utility.Vector3dVector(filtered_colors)

print(f"过滤后点云点数: {len(filtered_points)}")
print(f"删除的点数: {len(points) - len(filtered_points)}")

# 保存过滤后的点云
output_dir = os.path.dirname(pcd_path)
base_name = os.path.basename(pcd_path)
filtered_pcd_path = os.path.join(output_dir, f"filtered_{base_name}")
o3d.io.write_point_cloud(filtered_pcd_path, filtered_pcd)
print(f"已保存过滤后的点云到: {filtered_pcd_path}")

# 运行可视化
print("按下ESC键退出可视化窗口...")
vis.run()
vis.destroy_window()

# 可视化原始点云和过滤后的点云对比
vis_compare = o3d.visualization.Visualizer()
vis_compare.create_window(window_name="对比: 原始点云(白色) vs 过滤后点云(蓝色)")

# 添加原始点云（白色）
orig_pcd_copy = o3d.geometry.PointCloud()
orig_pcd_copy.points = o3d.utility.Vector3dVector(points)
orig_pcd_copy.paint_uniform_color([1, 1, 1])  # 白色
vis_compare.add_geometry(orig_pcd_copy)

# 添加过滤后的点云（蓝色）
if len(filtered_points) > 0:  # 确保有点可以添加
    filtered_pcd_copy = o3d.geometry.PointCloud()
    filtered_pcd_copy.points = o3d.utility.Vector3dVector(filtered_points)
    filtered_pcd_copy.paint_uniform_color([0, 0, 1])  # 蓝色
    vis_compare.add_geometry(filtered_pcd_copy)

# 添加边界框
for objid, info in vehicles.items():
    # 获取物体尺寸（长、高、宽）
    lhw = np.array(info['lhw'][0])
    # 计算边界框顶点
    points = get_vertices(lhw)
    # 获取物体位姿变换矩阵(b2v)
    b2v = info['rt'][0]

    # 平移到中心点坐标
    points_v = (b2v[:3, :3] @ points.T).T+b2v[:3, 3]

    connections = [[0, 1], [0, 2], [1, 3], [2, 3], [4, 5], [4, 6], [5, 7], [6, 7],
                   [1, 4], [0, 5], [3, 6], [2, 7]]

    colors = [[1, 0, 0] for _ in range(len(connections))]  # 红色边框

    # 创建线框模型
    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(points_v)
    line_set.lines = o3d.utility.Vector2iVector(connections)
    line_set.colors = o3d.utility.Vector3dVector(colors)

    # 将边界框添加到可视化窗口
    vis_compare.add_geometry(line_set)

render_opt_compare = vis_compare.get_render_option()
render_opt_compare.point_size = 2.0
vis_compare.run()
vis_compare.destroy_window()

print(f"已完成! 过滤后的点云已保存到: {filtered_pcd_path}")