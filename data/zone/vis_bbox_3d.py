import open3d as o3d
import numpy as np
import json
from scipy.spatial.transform import Rotation as SCR  # 用于旋转矩阵计算
from datetime import datetime  # 用于时间戳解析

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


# 读取PCD文件
pcd_path = "/home/sczone/hugsim_workspace/datasets/zone/3D_data_LSJWK4095NS119733/20250304_154844_0/Lidar/Pandar128_Compensate/n000001_2025-03-04-15-48-44-450027_Pandar128.pcd"  # 替换为你的PCD文件路径
json_file = "/home/sczone/hugsim_workspace/datasets/zone/3D_data_LSJWK4095NS119733/20250304_154844_0/annotations/trackOD/20250304_154844_0_1/cache/autolabel_4dod/20250304_154844_0_1/offline_showformat/n000001_2025-03-04-15-48-44-450027_Pandar128.pcd.json"

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
# 运行可视化
vis.run()
vis.destroy_window()
# 可视化点云
# o3d.visualization.draw_geometries([pcd])

print("end...")
