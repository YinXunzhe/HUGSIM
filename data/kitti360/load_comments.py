import os
import numpy as np
import matplotlib.pyplot as plt
from imageio.v2 import imread, imwrite
from glob import glob
from tqdm import tqdm
import cv2
import json
import argparse
from utils.fish2persp.get_virtual_perspective import virtual_perspect
from kitti360.annotation import Annotation3D
import math

def fov2focal(fov, pixels):
    """将视场角(FOV)转换为焦距
    Args:
        fov: 视场角（弧度）
        pixels: 传感器尺寸（像素数）
    Returns:
        焦距（像素单位）
    """
    return pixels / (2 * math.tan(fov / 2))

def loadCameraToPose(filename):
    """加载相机到全局坐标系的变换矩阵
    Args:
        filename: 标定文件路径
    Returns:
        三个相机的变换矩阵（image_01, image_02, image_03）
    """
    Tr = {}
    lastrow = np.array([0, 0, 0, 1]).reshape(1, 4)  # 齐次坐标最后一行
    with open(filename, 'r') as f:
        lines = f.readlines()
        for line in lines:
            lineData = list(line.strip().split())
            data = np.array(lineData[1:]).reshape(3,4).astype(np.float64)
            data = np.concatenate((data,lastrow), axis=0)  # 转换为4x4齐次矩阵
            Tr[lineData[0]] = data
    return Tr['image_01:'], Tr['image_02:'], Tr['image_03:']

def load_data(datadir, sequence='2013_05_28_drive_0000_sync'):
    """加载相机内外参数据
    Args:
        datadir: 数据根目录
        sequence: 序列名称
    Returns:
        各相机的位姿矩阵和内参矩阵
    """
    # 加载内参矩阵
    intrinstic_file = os.path.join(os.path.join(datadir, 'calibration'), 'perspective.txt')
    with open(intrinstic_file) as f:
        lines = f.readlines()
        for line in lines:
            lineData = line.strip().split()
            if lineData[0] == 'P_rect_00:':
                K_00 = np.array(lineData[1:]).reshape(3,4).astype(np.float64)
                K_00 = K_00[:,:-1]  # 移除最后一列
            elif lineData[0] == 'P_rect_01:':
                K_01 = np.array(lineData[1:]).reshape(3,4).astype(np.float64)
                K_01 = K_01[:,:-1]
            elif lineData[0] == 'R_rect_01:':
                R_rect_01 = np.eye(4)
                R_rect_01[:3,:3] = np.array(lineData[1:]).reshape(3,3).astype(np.float64)

    # 加载外参矩阵
    CamPose_00 = {}
    extrinstic_file = os.path.join(datadir,os.path.join('data_poses',sequence))
    cam2world_file_00 = os.path.join(extrinstic_file,'cam0_to_world.txt')
    
    # 加载相机00到世界坐标系的变换
    with open(cam2world_file_00,'r') as f:
        lines = f.readlines()
        for line in lines:
            lineData = list(map(float,line.strip().split()))
            CamPose_00[int(lineData[0])] = np.array(lineData[1:]).reshape(4,4)
    
    # 加载其他相机的变换
    CamToPose_01, CamToPose_02, CamToPose_03 = loadCameraToPose(os.path.join(os.path.join(datadir, 'calibration'),'calib_cam_to_pose.txt'))
    poses = np.loadtxt(os.path.join(extrinstic_file,'poses.txt'))
    frames = poses[:, 0]
    poses = np.reshape(poses[:, 1:], [-1, 3, 4])
    
    # 计算各相机的位姿矩阵
    for frame, pose in zip(frames, poses):
        pose = np.concatenate((pose, np.array([0., 0., 0., 1.]).reshape(1, 4)))
        pp = np.matmul(pose, CamToPose_01)
        CamPose_01[int(frame)] = np.matmul(pp, np.linalg.inv(R_rect_01))
        CamPose_02[int(frame)] = np.matmul(pose, CamToPose_02)
        CamPose_03[int(frame)] = np.matmul(pose, CamToPose_03)
    
    # 构建齐次形式的内参矩阵
    hom_K00, hom_K01 = np.eye(4), np.eye(4)
    hom_K00[:3, :3] = K_00
    hom_K01[:3, :3] = K_01

    return CamPose_00, CamPose_01, CamPose_02, CamPose_03, hom_K00, hom_K01, None, None

def get_kitti360_bbox(datadir, seq, start_index, end_index, inv_pose):
    """加载3D边界框数据
    Args:
        datadir: 数据目录
        seq: 序列名称
        start_index/end_index: 时间范围
        inv_pose: 逆位姿矩阵（用于坐标系转换）
    Returns:
        rts: 各时刻物体的变换矩阵
        verts: 各物体的顶点坐标
    """
    annotation3D = Annotation3D(datadir, seq)
    rts = {}
    verts = {}
    # 标准立方体顶点（局部坐标系）
    cano_verts = np.array([[0.5, 0.5, 0.5], [0.5, 0.5, -0.5], [0.5, -0.5,  0.5], [0.5, -0.5, -0.5],
                           [-0.5, 0.5, -0.5], [-0.5, 0.5, 0.5], [-0.5, -0.5, -0.5], [-0.5, -0.5, 0.5]])
    
    # 遍历所有标注对象
    for _, annotation in annotation3D.objects.items():
        for timestamp in annotation.keys():
            if timestamp < 0 or not (start_index <= timestamp < end_index):
                continue
            t = timestamp - start_index
            obj = annotation[timestamp]
            # 只保留特定类别的物体（26-28为车辆类）
            if not (obj.semanticId == 26 or obj.semanticId == 27 or obj.semanticId == 28):
                continue
            
            track_id = int(obj.instanceId)
            R = obj.R 
            bsize = np.linalg.norm(R, axis=0)  # 边界框尺寸
            vertices = cano_verts * bsize      # 缩放顶点
            R = R / bsize                      # 归一化旋转矩阵
            T = obj.T                          # 平移向量
            
            # 构建物体到世界坐标系的变换矩阵
            P = np.eye(4)
            P[:3, :3] = R
            P[:3, 3] = T
            B2W = np.dot(inv_pose, P)  # 转换到中间坐标系
            
            # 存储结果
            if t not in rts:
                rts[t] = {}
            rts[t][track_id] = B2W.tolist()
            if track_id not in verts:
                verts[track_id] = vertices.tolist()

    return rts, verts

def get_opts():
    """解析命令行参数"""
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=str, required=True, help='数据根目录')
    parser.add_argument('--out', type=str, required=True, help='输出目录')
    parser.add_argument('--start', type=int, required=True, help='起始帧')
    parser.add_argument('--end', type=int, required=True, help='结束帧')
    parser.add_argument('--cams', nargs='+', type=int, required=True, help='使用的相机列表')
    return parser.parse_args()

def read_smt_package(out, cam, fn):
    """读取语义分割数据包"""
    smt_path = os.path.join(out, 'semantics', f'cam_{cam}_fisheye', fn)
    return {
        'smt': np.load(smt_path + '.npy'),     # 语义分割数据
        'comp': imread(smt_path + '_comp.png'),# 合成图像
        'vis': imread(smt_path + '_vis.png'),  # 可视化图像
    }
    
def save_smt_package(out, cam, fn, save_package):
    """保存语义分割数据包"""
    smt_path = os.path.join(out, 'semantics', f'cam_{cam}', fn)
    np.save(smt_path+'.npy', save_package['smt'])
    imwrite(smt_path+'_comp.png', save_package['comp'].astype(np.uint8))
    imwrite(smt_path+'_vis.png', save_package['vis'].astype(np.uint8))

def fish_eye(c2w, H, W, vk, out, cam, fn):
    """处理鱼眼相机图像到透视投影的转换
    Args:
        c2w: 相机到世界坐标系的变换矩阵
        H/W: 输出图像高宽
        vk: 虚拟相机内参
        out: 输出目录
        cam: 相机编号
        fn: 文件名
    Returns:
        更新后的相机参数
    """
    img = cv2.imread(os.path.join(out, 'images', f'cam_{cam}_fisheye', f'{fn}.png'))
    smt_package = read_smt_package(out, cam, fn)
    
    # 根据相机位置选择处理方式
    if cam == '2':
        imgV, smtV_package, mask = virtual_perspect(H, W, vk, img, smt_package=smt_package, left=True)
    else:
        imgV, smtV_package, mask = virtual_perspect(H, W, vk, img, smt_package=smt_package, left=False)
    
    # 保存处理结果
    cv2.imwrite(os.path.join(out, 'images', f'cam_{cam}', f'{fn}.png'), imgV)
    save_smt_package(out, cam, fn, smtV_package)
    return c2w, vk, H, W

if __name__ == '__main__':
    # 解析命令行参数
    args = get_opts()
    datadir = args.root
    output_dir = args.out
    start = args.start
    end = args.end
    
    # 加载相机参数
    p0, p1, p2, p3, k0, k1, _, _ = load_data(datadir, sequence='2013_05_28_drive_0000_sync')

    # 初始化元数据
    meta_data = {
        "camera_model": "OPENCV",
        "frames": [],
    }

    # 计算参考帧的逆变换矩阵
    frames = sorted(glob(os.path.join(output_dir, 'images', 'cam_0', '*.png')))
    frames_name = [os.path.basename(f).split('.')[0] for f in frames]
    inv_pose = np.linalg.inv(p0[int(frames_name[0])])
    meta_data['inv_pose'] = inv_pose.tolist()
    meta_data['ref_frame'] = frames_name[0]

    # 加载3D边界框信息
    rts, verts = get_kitti360_bbox(os.path.join(datadir, 'data_3d_bboxes'), 
                                  '2013_05_28_drive_0000_sync', start, end, inv_pose)
    meta_data['verts'] = verts

    # 设置虚拟相机参数
    H, W = 360, 600
    fovx, fovy = 0.7 * np.pi, 0.6 * np.pi
    fx = fov2focal(fovx, W)
    fy = fov2focal(fovy, H)
    vk = np.array([[fx, 0, W//2, 0],
                   [0, fy, H//2, 0],
                   [0, 0,   1,   0],
                   [0, 0,   0,   1]])

    # 处理所有帧数据
    available_cams = [f'cam_{cam}' for cam in args.cams]
    for fn in tqdm(frames_name):
        for cam in available_cams:
            # 根据相机类型处理数据
            if cam == 'cam_0':
                c2w = p0[int(fn)]
                k = k0
                h, w = 376, 1408
            elif cam == 'cam_1':
                c2w = p1[int(fn)]
                k = k1
                h, w = 376, 1408
            elif cam == 'cam_2':
                c2w = p2[int(fn)]
                c2w, k, h, w = fish_eye(c2w, H, W, vk, output_dir, '2', fn)
            elif cam == 'cam_3':
                c2w = p3[int(fn)]
                c2w, k, h, w = fish_eye(c2w, H, W, vk, output_dir, '3', fn)
            else:
                raise NotImplementedError
            
            # 转换到参考坐标系
            c2w = np.dot(inv_pose, c2w)
            
            # 构建帧信息
            meta_data['frames'].append({
                "rgb_path": os.path.join('./images', cam, f'{fn}.png'),
                "camtoworld": c2w.tolist(),
                "intrinsics": k.tolist(),
                "width": w,
                "height": h,
                "timestamp": (int(fn) - start) * 0.1,  # 转换为秒
                "dynamics": rts.get(int(fn) - start, {})  # 动态物体信息
            })
    
    # 保存元数据
    with open(os.path.join(output_dir, 'meta_data.json'), 'w') as out_file:
        json.dump(meta_data, out_file, indent=4)
