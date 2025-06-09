import numpy as np
from datetime import datetime  # 用于时间戳解析

AVAILABLE_CAMERAS = (
    "CAM_FRONT_120",
    "CAM_FRONT_LEFT",
    "CAM_FRONT_RIGHT",
    "CAM_BACK",
    "CAM_BACK_LEFT",
    "CAM_BACK_RIGHT",
)

def get_rgb_by_distance(cur_val, min_val=0, max_val=50):
    jet_color_matrix = [0, 0, 0.5625,
                        0, 0, 0.6250,
                        0, 0, 0.6875,
                        0, 0, 0.7500,
                        0, 0, 0.8125,
                        0, 0, 0.8750,
                        0, 0, 0.9375,
                        0, 0, 1,
                        0, 0.0625, 1,
                        0, 0.1250, 1,
                        0, 0.1875, 1,
                        0, 0.2500, 1,
                        0, 0.3125, 1,
                        0, 0.3750, 1,
                        0, 0.4375, 1,
                        0, 0.5000, 1,
                        0, 0.5625, 1,
                        0, 0.6250, 1,
                        0, 0.6875, 1,
                        0, 0.7500, 1,
                        0, 0.8125, 1,
                        0, 0.8750, 1,
                        0, 0.9375, 1,
                        0, 1, 1,
                        0.0625, 1, 0.9375,
                        0.1250, 1, 0.8750,
                        0.1875, 1, 0.8125,
                        0.2500, 1, 0.7500,
                        0.3125, 1, 0.6875,
                        0.3750, 1, 0.6250,
                        0.4375, 1, 0.5625,
                        0.5000, 1, 0.5000,
                        0.5625, 1, 0.4375,
                        0.6250, 1, 0.3750,
                        0.6875, 1, 0.3125,
                        0.7500, 1, 0.2500,
                        0.8125, 1, 0.1875,
                        0.8750, 1, 0.1250,
                        0.9375, 1, 0.0625,
                        1, 1, 0,
                        1, 0.9375, 0,
                        1, 0.8750, 0,
                        1, 0.8125, 0,
                        1, 0.7500, 0,
                        1, 0.6875, 0,
                        1, 0.6250, 0,
                        1, 0.5625, 0,
                        1, 0.5000, 0,
                        1, 0.4375, 0,
                        1, 0.3750, 0,
                        1, 0.3125, 0,
                        1, 0.2500, 0,
                        1, 0.1875, 0,
                        1, 0.1250, 0,
                        1, 0.0625, 0,
                        1, 0, 0,
                        0.9375, 0, 0,
                        0.8750, 0, 0,
                        0.8125, 0, 0,
                        0.7500, 0, 0,
                        0.6875, 0, 0,
                        0.6250, 0, 0,
                        0.5625, 0, 0,
                        0.5000, 0, 0]
    jet_color_matrix = np.reshape(np.asarray(jet_color_matrix) * 255, (64, 3))
    jet_color_matrix = jet_color_matrix.astype(dtype=np.uint8)

    cur_val = np.clip(cur_val, min_val, max_val)
    index = (cur_val - min_val) / max_val * 63
    index = np.round(index).astype(np.int8)
    rgb_val = jet_color_matrix[index, :]

    return rgb_val

# 为了和vertex里的坐标顺序为lhw匹配
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

def parse_timestamp(filename):
    """从文件名解析时间戳，格式为YYYY-MM-DD-HH-MM-SS-ffffff"""
    # 提取时间戳部分（假设文件名结构为prefix_timestamp_suffix）
    timestamp_str = filename.split('_')[1]
    # 分割时间戳的各个部分
    parts = timestamp_str.split('-')
    if len(parts) != 7:
        raise ValueError(f"无效的时间戳格式: {timestamp_str}")
    # 组合成datetime可解析的字符串格式
    dt_str = f"{parts[0]}-{parts[1]}-{parts[2]} {parts[3]}:{parts[4]}:{parts[5]}.{parts[6]}"
    # 解析为datetime对象
    dt = datetime.strptime(dt_str, "%Y-%m-%d %H:%M:%S.%f")
    return dt.timestamp()