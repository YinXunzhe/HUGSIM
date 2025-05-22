"""
HUGSIM 地面模型训练主程序
文件路径：/home/sczone/hugsim_workspace/HUGSIM/train_ground.py
功能：实现地面点云模型的高斯泼溅（Gaussian Splatting）训练流程
"""

# 基础库导入
import os
import torch
import sys
import uuid
import json
from argparse import ArgumentParser
from functools import partial

# 自定义工具库
from utils.loss_utils import l1_loss, ssim_loss  # 损失函数
from utils.dataset import HUGSIM_dataset, hugsim_collate, tocuda  # 数据加载

# 3D渲染相关
from gaussian_renderer import render_ground  # 高斯渲染器
from scene.ground_model import GroundModel  # 地面模型定义
from scene import load_cameras  # 相机参数加载
from scene.dataset_readers import fetchPly  # 点云数据读取

# 训练指标
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

# 配置管理
from omegaconf import OmegaConf

# 进度条显示
from tqdm import tqdm as std_tqdm
tqdm = partial(std_tqdm, dynamic_ncols=True)  # 自定义进度条格式

# 初始化全局结果字典
results = {'train': {}, 'test': {}}

# 初始化评估指标（CUDA设备）
m_psnr = PeakSignalNoiseRatio(data_range=1.0).to('cuda')
m_ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to('cuda')
m_lpips = LearnedPerceptualImagePatchSimilarity().to('cuda')

def training(cfg):
    """主要训练流程
    Args:
        cfg: 配置对象，包含训练参数和路径配置
    """
    # 初始化语义分割损失（如果启用）
    if cfg.semantic:
        semantic_ce = CrossEntropyLoss()

    # 加载相机数据并创建数据集
    train_cams, test_cams, _ = load_cameras(cfg, cfg.data_type, True)
    train_dataset = HUGSIM_dataset(train_cams, cfg.data_type)
    test_dataset = HUGSIM_dataset(test_cams, cfg.data_type)
    
    # 创建数据加载器（batch_size=1 用于逐样本处理）
    train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True, 
                                pin_memory=True, collate_fn=hugsim_collate)
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False,
                               pin_memory=True, collate_fn=hugsim_collate)

    # 初始化输出目录
    prepare_output(cfg)
    
    # 加载点云数据并初始化地面模型
    pcd = fetchPly(os.path.join(cfg.source_path, 'ground_points3d.ply'))
    gaussians = GroundModel(cfg.model.sh_degree, pcd)

    # 设置背景颜色（根据配置选择白色/黑色）
    bg_color = [1, 1, 1] if cfg.model.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    # 训练循环控制
    progress_bar = tqdm(range(first_iter, cfg.train.iterations), desc="Training progress")
    
    for iteration in range(first_iter, cfg.train.iterations + 1):
        # 球谐函数度数升级策略（每1000次迭代）
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        # 数据加载与预处理
        view_iid, prev_iid, gt_image, gt_semantic, gt_flow, gt_depth, mask = next(iter(train_dataloader))
        # 将数据移至GPU
        gt_image, gt_semantic = gt_image.cuda(), tocuda(gt_semantic)
        
        # 高斯渲染过程
        render_pkg = render_ground(viewpoint_cam, gaussians, background)
        image = render_pkg["render"]  # 渲染得到的RGB图像

        # 损失计算模块
        loss = 0
        # RGB重建损失（L1 + SSIM）
        Ll1 = l1_loss(image, gt_image)
        rgb_loss = (1.0 - cfg.opt.lambda_dssim) * Ll1 + cfg.opt.lambda_dssim * ssim_loss(image, gt_image)
        loss += rgb_loss

        # 3D结构一致性损失（防止地面点云扭曲）
        distort_3d_loss = calculate_3d_distortion(gaussians, viewpoint_cam, cfg)
        loss += distort_3d_loss

        # 反向传播与优化
        loss.backward()
        gaussians.optimizer.step()
        gaussians.optimizer.zero_grad()

        # 模型致密化策略（控制高斯分布密度）
        if iteration < cfg.opt.densify_until_iter:
            handle_densification(gaussians, render_pkg, cfg, iteration)

        # 验证与模型保存
        if iteration % cfg.train.checkpoint_interval == 0:
            validation(iteration, cfg.model_path, gaussians, train_cams, test_cams, 
                      render_ground, background)
            save_checkpoint(cfg, gaussians, iteration)

def validation(iteration, model_path, gaussians, train_cameras, test_cameras, renderFunc, background):
    """模型验证与指标计算
    Args:
        iteration: 当前迭代次数
        model_path: 模型保存路径
        gaussians: 地面模型实例
        train_cameras/test_cameras: 训练/测试相机数据
        renderFunc: 渲染函数
        background: 背景颜色配置
    """
    # 在训练集和测试集上评估
    validation_configs = (
        {'name': 'test', 'cameras': test_cameras},
        {'name': 'train', 'cameras': train_cameras}
    )
    
    for config in validation_configs:
        # 初始化指标记录器
        metrics = {'l1': 0, 'psnr': 0, 'ssim': 0, 'lpips': 0}
        
        for viewpoint in config['cameras']:
            # 渲染与真值获取
            gt_image = viewpoint.original_image.cuda()
            image = torch.clamp(renderFunc(viewpoint, gaussians, background)["render"], 0.0, 1.0)
            
            # 计算各项质量指标
            metrics['l1'] += l1_loss(image, gt_image).mean()
            metrics['psnr'] += m_psnr(image[None, ...], gt_image[None, ...])
            metrics['ssim'] += m_ssim(image[None, ...], gt_image[None, ...])
            metrics['lpips'] += m_lpips(image[None, ...], gt_image[None, ...])

        # 保存结果并打印
        save_metrics(iteration, metrics, config['name'])
        print_metrics(iteration, metrics, config['name'])
    
    # 将结果写入JSON文件
    with open(os.path.join(model_path, 'ground', 'results.json'), 'w') as wf:
        json.dump(results, wf, indent=4)

def main():
    """主函数：配置加载与训练初始化"""
    # 命令行参数解析
    parser = ArgumentParser(description="Training script parameters")
    parser.add_argument("--base_cfg", type=str, default="./configs/gs_base.yaml")
    parser.add_argument("--data_cfg", type=str, default="./configs/nusc.yaml")
    parser.add_argument("--source_path", type=str, default="")
    parser.add_argument("--model_path", type=str, default="")
    args = parser.parse_args()
    
    # 合并配置文件
    cfg = OmegaConf.merge(OmegaConf.load(args.base_cfg), OmegaConf.load(args.data_cfg))
    
    # 路径覆盖逻辑
    if args.source_path:
        cfg.source_path = args.source_path
    if args.model_path:
        cfg.model_path = args.model_path
        
    # 启动训练流程
    print("Optimizing " + args.model_path)
    training(cfg)
    print("\nTraining complete.")

if __name__ == "__main__":
    main()
