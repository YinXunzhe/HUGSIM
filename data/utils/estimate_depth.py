import argparse
import glob
import os
import json
import numpy as np
import torch
from tqdm import tqdm
from unidepth.models import UniDepthV2
from PIL import Image
import cv2
import matplotlib.pyplot as plt

def colorize_depth(depth, colormap=cv2.COLORMAP_JET, valid_range=None):
    """
    将深度图转换为彩色可视化图像
    Args:
        depth (np.ndarray): 深度图 (H,W)
        colormap: OpenCV颜色映射 (默认JET)
        valid_range: 深度有效范围 (min, max)
    Returns:
        np.ndarray: 彩色深度图 (H,W,3)
    """
    if valid_range is None:
        valid_mask = np.ones_like(depth, dtype=bool)
    else:
        valid_mask = (depth >= valid_range[0]) & (depth <= valid_range[1])
    
    # 归一化有效区域
    depth_normalized = np.zeros_like(depth, dtype=np.float32)
    if valid_mask.any():
        depth_valid = depth[valid_mask]
        min_d, max_d = depth_valid.min(), depth_valid.max()
        depth_normalized[valid_mask] = (depth_valid - min_d) / (max_d - min_d + 1e-6) * 255
    
    # 应用颜色映射
    depth_colored = cv2.applyColorMap(depth_normalized.astype(np.uint8), colormap)
    
    # 标记无效区域为黑色
    depth_colored[~valid_mask] = 0
    return depth_colored

def get_opts():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=str, required=True)
    parser.add_argument('--save_img', type=str, default=False)
    # 当深度值普遍较小时，降低scale值可使颜色分布更明显
    parser.add_argument('--vis_scale', type=float, default=1.0, 
                       help="Scale factor for visualization (adjust for better color range)")    
    return parser.parse_args()

if __name__ == '__main__':
    args = get_opts()
    
    print('loading depth model...')
    model = UniDepthV2.from_pretrained("lpiccinelli/unidepth-v2-vitl14", force_download=False)
    # model = UniDepthV2.from_pretrained("/workspace/model_zoo/unidepth/unidepth-v2-vits14")
    model = model.to("cuda")
    model.eval()
    print("Depth model loaded")
    
    os.makedirs(os.path.join(args.out, 'depth'), exist_ok=True)
    for cam_pth in glob.glob(os.path.join(args.out, 'images', '*')):
        cam = os.path.basename(cam_pth)
        os.makedirs(os.path.join(args.out, 'depth', cam), exist_ok=True)
    
    with open(os.path.join(args.out, 'meta_data.json')) as f:
        meta_data = json.load(f)
    
    for frame in tqdm(meta_data['frames']):
        im_path = os.path.join(args.out, frame['rgb_path'])
        K = np.array(frame['intrinsics'])
        K = torch.from_numpy(K[:3, :3]).float().cuda()
        image = torch.from_numpy(np.array(Image.open(im_path))).permute(2, 0, 1)
        prediction = model.infer(image, K)
        depth = prediction["depth"][0][0].detach().cpu()  # Depth in [m].
        
        depth_path = os.path.join(
            args.out,
            im_path.replace("images", "depth")
            .replace("./", "")
            .replace(".jpg", ".pt")
            .replace(".png", ".pt"),
        )
        
        torch.save(depth, depth_path)
        
        if args.save_img:

            # 可视化深度图保存路径
            vis_path = depth_path.replace('.pt', '.png')
            depth_np = depth.numpy()
            
            # 调整可视化范围 (可选)
            depth_scaled = depth_np * args.vis_scale
            
            # 生成彩色深度图
            depth_colored = colorize_depth(depth_scaled)
            
            # 保存可视化结果
            cv2.imwrite(vis_path, cv2.cvtColor(depth_colored, cv2.COLOR_RGB2BGR))