"""
3D高斯溅射训练主程序 - 用于动态场景重建和语义分割
实现包含数据加载、模型训练、验证及结果保存等功能
"""

import os
import torch
from utils.loss_utils import l1_loss, ssim, ssim_loss  # 自定义损失函数
from gaussian_renderer import render  # 高斯渲染器
from scene import Scene, GaussianModel  # 场景和高斯模型类
import uuid
from torchmetrics.image import PeakSignalNoiseRatio  # 图像质量评估指标
from torchmetrics.image import StructuralSimilarityIndexMeasure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from argparse import ArgumentParser  # 命令行参数解析
from torch.nn import CrossEntropyLoss  # 交叉熵损失
import json
import pickle
import torchvision
from utils.dataset import HUGSIM_dataset, hugsim_collate, tocuda  # 自定义数据集类
from omegaconf import OmegaConf  # 配置管理
from torch.utils.data import DataLoader  # 数据加载
from functools import partial
from tqdm import tqdm as std_tqdm  # 进度条
from utils.dynamic_utils import create_unicycle_model  # 动态模型创建

# 初始化带动态调整的进度条
tqdm = partial(std_tqdm, dynamic_ncols=True)

# 结果存储字典，包含训练和测试指标
results = {'train': {}, 'test': {}}

# 初始化评估指标（使用GPU加速）
m_psnr = PeakSignalNoiseRatio(data_range=1.0).to('cuda')
m_ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to('cuda')
m_lpips = LearnedPerceptualImagePatchSimilarity().to('cuda')

def training(cfg):
    """主训练函数，包含完整的训练流程"""
    
    # 语义分割相关初始化
    if cfg.semantic:
        semantic_ce = CrossEntropyLoss()
    
    # 加载相机位姿参数
    with open(os.path.join(cfg.source_path, 'ground_param.pkl'), 'rb') as f:
        cam_poses, _, _ = pickle.load(f)
        cam_positions = torch.tensor(cam_poses[:, :3, 3]).float().cuda()

    first_iter = 0
    prepare_output(cfg)  # 准备输出目录
    
    # 初始化高斯模型和场景
    (ground_model_params, _) = torch.load(os.path.join(cfg.model_path, "ckpts", f"ground_chkpnt30000.pth"))
    gaussians = GaussianModel(cfg.model.sh_degree, 
                             feat_mutable=True,
                             affine=cfg.affine,
                             ground_args=ground_model_params)
    scene = Scene(cfg, gaussians, data_type=cfg.data_type)
    
    # 设置优化器
    scene.gaussians.training_setup(cfg.opt)
    for iid, dynamic_gaussian in scene.dynamic_gaussians.items():
        dynamic_gaussian.training_setup(cfg.opt)
    
    # 初始化动态单目模型
    if cfg.unicycle:
        unicycles = create_unicycle_model(scene.getTrainCameras(), cfg.model_path,
                                        cfg.uc_fit_iter, cfg.uc_opt_pos, cfg.data_type)
    else:
        unicycles = {}

    # 背景颜色设置
    bg_color = [1, 1, 1] if cfg.model.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    # CUDA事件计时器
    iter_start = torch.cuda.Event(enable_timing=True)
    iter_end = torch.cuda.Event(enable_timing=True)

    # 训练进度条初始化
    progress_bar = tqdm(range(first_iter, cfg.train.iterations), desc="Training progress")
    first_iter += 1

    # 创建训练结果保存目录
    os.makedirs(os.path.join(scene.model_path, "save_train"), exist_ok=True)

    # 数据加载器配置
    train_cams = scene.getTrainCameras().copy()
    train_dataset = HUGSIM_dataset(train_cams, cfg.data_type)
    train_dataloader = DataLoader(train_dataset, batch_size=1, 
                                shuffle=True, pin_memory=True, collate_fn=hugsim_collate)

    # 主训练循环
    for iteration in range(first_iter, cfg.train.iterations + 1):        
        iter_start.record()

        # 学习率更新
        scene.gaussians.update_learning_rate(iteration)
        for iid, dynamic_gaussian in scene.dynamic_gaussians.items():
            dynamic_gaussian.update_learning_rate(iteration)

        # 每1000次迭代提升球谐阶数
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()
            for iid, dynamic_gaussian in scene.dynamic_gaussians.items():
                dynamic_gaussian.oneupSHdegree()

        # 获取训练数据
        view_iid, prev_iid, gt_image, gt_semantic, gt_flow, gt_depth, mask = next(iter(train_dataloader))
        gt_image, gt_flow, gt_depth, mask = gt_image.cuda(), tocuda(gt_flow), tocuda(gt_depth), tocuda(mask)
        viewpoint_cam = train_cams[view_iid]

        # 渲染图像
        render_pkg = render(viewpoint_cam, None, gaussians, scene.dynamic_gaussians, unicycles, background)
        
        # 提取渲染结果
        image, viewspace_point_tensor, info = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["info"]
        radii = info["radii"][0]
        visibility_filter = radii > 0
        viewspace_point_tensor.retain_grad()

        # 定期保存训练图像
        if iteration % 500 == 0:
            torchvision.utils.save_image(image, os.path.join(scene.model_path, "save_train", f"{iteration}_{viewpoint_cam.image_name}.png"))

        # 损失计算
        loss = 0

        # 语义分割损失
        if cfg.semantic and gt_semantic is not None:
            gt_semantic = gt_semantic.cuda()
            semantic_map = render_pkg["feats"]
            if mask is not None and cfg.ignore_dynamic:
                gt_semantic[:, ~mask] = torch.argmax(semantic_map[:, ~mask], dim=0).detach()
            semantic_loss = semantic_ce(semantic_map.permute(1,2,0).view(-1, 20), gt_semantic.view(-1)) * 0.01
            loss += semantic_loss

        # 动态区域掩码处理
        if mask is not None and cfg.ignore_dynamic:
            gt_image[:, ~mask] = image[:, ~mask].detach()
            
        # RGB重建损失
        Ll1 = l1_loss(image, gt_image)
        rgb_loss = (1.0 - cfg.opt.lambda_dssim) * Ll1 + cfg.opt.lambda_dssim * ssim_loss(image, gt_image)
        loss += rgb_loss

        # 3D形变正则化损失
        distort_3d_loss = 0 
        N_sample=10
        grid_length = 0.2
        if iteration > 2000:
            # 地面点筛选和坐标转换
            ground_mask = torch.argmax(gaussians.get_3D_features, dim=1) <= 1
            w2c = torch.linalg.inv(viewpoint_cam.c2w)
            points = gaussians.get_xyz[ground_mask]
            c_points = (w2c[:3, :3] @ points.T).T + w2c[:3, 3]
            points_gd = gaussians.ground_model.get_xyz
            c_points_gd = (w2c[:3, :3] @ points_gd.T).T + w2c[:3, 3]
            
            # 分层采样计算形变损失
            biases = -2 + 4 * torch.rand(N_sample, device='cuda')
            for bias in biases:
                mask = (bias < c_points[:, 2]) & (c_points[:, 2] < (bias + grid_length)) 
                if torch.sum(mask) == 0:
                    continue
                ys = c_points[mask, 1]
                mask_gd = (bias < c_points_gd[:, 2]) & (c_points_gd[:, 2] < (bias + grid_length)) 
                ys_gd = c_points_gd[mask_gd, 1]
                distort_3d_loss += torch.mean((ys - torch.mean(ys_gd))**2)
            distort_3d_loss /= N_sample
            loss += distort_3d_loss

        # 单目模型正则化损失
        reg_loss = 0
        if cfg.uc_opt_pos and (len(unicycles) > 0) and (1000 < iteration) and (iteration < 15000):
            for track_id, unicycle_pkg in unicycles.items():
                model = unicycle_pkg['model']
                reg_loss += 1e-3 * model.reg_loss() + 1e-4 * model.pos_loss()
            reg_loss = reg_loss / len(unicycles)
            loss += reg_loss

        # 反向传播
        loss.backward()

        iter_end.record()

        with torch.no_grad():
            # 更新进度条
            if iteration % 10 == 0:
                postfix = {"RGB": f"{rgb_loss:.{4}f}"}
                if cfg.semantic:
                    postfix["Semantic"] = f"{semantic_loss:.{4}f}"
                if reg_loss != 0:
                    postfix["UniReg"] = f"{reg_loss:.{4}f}"
                if distort_3d_loss != 0:
                    postfix['dist3d'] = f"{distort_3d_loss:.{4}f}"
                progress_bar.set_postfix(postfix)
                progress_bar.update(10)
            if iteration == cfg.train.iterations:
                progress_bar.close()

            # 模型保存和验证
            torch.cuda.synchronize()
            
            if (iteration in cfg.train.checkpoint_iterations):
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                os.makedirs(scene.model_path + '/ckpts', exist_ok=True)
                
                # 保存主要模型参数
                torch.save((gaussians.capture(), iteration), 
                         os.path.join(scene.model_path, f"ckpts/chkpnt{str(iteration)}.pth"))

                # 保存动态模型参数
                for iid, dynamic_gaussian in scene.dynamic_gaussians.items():
                    torch.save((dynamic_gaussian.capture(), iteration), 
                             os.path.join(scene.model_path, f"ckpts/dynamic_{iid}_chkpnt{iteration}.pth"))
                
                # 保存单目模型参数
                for track_id, unicycle_pkg in unicycles.items():
                    model = unicycle_pkg['model']
                    torch.save(model.capture(), 
                             os.path.join(scene.model_path, f"ckpts/unicycle_{track_id}.pth"))
                    model.visualize(os.path.join(scene.model_path, "unicycle", f"{track_id}_{iteration}.png"))

                validation(iteration, scene, render, [scene.dynamic_gaussians, unicycles, background])

                # 保存完整场景
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)

            # 参数优化步骤
            if iteration < cfg.train.iterations:
                # 主模型优化
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none=True)
                if gaussians.ground_optimizer is not None:
                    gaussians.ground_optimizer.step()
                    gaussians.ground_optimizer.zero_grad(set_to_none=True)

                # 动态模型优化
                for iid, dynamic_gaussian in scene.dynamic_gaussians.items():
                    dynamic_gaussian.optimizer.step()
                    dynamic_gaussian.optimizer.zero_grad(set_to_none=True)

                # 单目模型优化
                if cfg.unicycle and cfg.uc_opt_pos and iteration > 1000:
                    for track_id, unicycle_pkg in unicycles.items():
                        unicycle_optimizer = unicycle_pkg['optimizer']
                        unicycle_optimizer.step()
                        unicycle_optimizer.zero_grad(set_to_none=True)

            # 点云密度调整
            if iteration < cfg.opt.densify_until_iter:
                # 梯度统计和可见性过滤
                grad = viewspace_point_tensor.grad[0].clone()
                grad[..., 0] *= info['width'] / 2.0
                grad[..., 1] *= info['height'] / 2.0
                
                # 主模型密度调整
                current_index = gaussians.get_xyz.shape[0]
                gaussians.max_radii2D[visibility_filter[:current_index]] = torch.max(
                    gaussians.max_radii2D[visibility_filter[:current_index]], 
                    radii[:current_index][visibility_filter[:current_index]]
                )
                gaussians.add_densification_stats_grad(grad[:current_index], visibility_filter[:current_index])
                last_index = current_index

                # 动态模型密度调整
                for iid in viewpoint_cam.dynamics.keys():
                    dynamic_gaussian = scene.dynamic_gaussians[iid]
                    current_index = last_index + dynamic_gaussian.get_xyz.shape[0]
                    visible_mask = visibility_filter[last_index:current_index]
                    dynamic_gaussian.max_radii2D[visible_mask] = torch.max(
                        dynamic_gaussian.max_radii2D[visible_mask],
                        radii[last_index:current_index][visible_mask]
                    )
                    dynamic_gaussian.add_densification_stats_grad(grad[last_index:current_index], visible_mask)
                    last_index = current_index

                # 定期进行密度优化和剪枝
                if iteration > cfg.opt.densify_from_iter and iteration % cfg.opt.densification_interval == 0:
                    size_threshold = 20 if iteration > cfg.opt.opacity_reset_interval else None
                    gaussians.densify_and_prune(cfg.opt.densify_grad_threshold, 0.005, 
                                              scene.cameras_extent, size_threshold, cam_pos=cam_positions)

                # 透明度重置
                if iteration % cfg.opt.opacity_reset_interval == 0 or (cfg.model.white_background and iteration == cfg.opt.densify_from_iter):
                    gaussians.reset_opacity()

def prepare_output(args):    
    """输出目录准备函数"""
    if not args.model_path:
        # 生成唯一实验ID
        unique_str = os.getenv('OAR_JOB_ID', str(uuid.uuid4()))
        args.model_path = os.path.join("./output/", unique_str[0:10])
        
    # 创建相关目录
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok=True)
    os.makedirs(os.path.join(args.model_path, "unicycle"), exist_ok=True)

def validation(iteration, scene, renderFunc, renderArgs):
    """验证函数，计算图像质量指标并保存结果"""
    os.makedirs(os.path.join(scene.model_path, "save_test"), exist_ok=True)

    # 配置验证集和训练子集
    torch.cuda.empty_cache()
    validation_configs = (
        {'name': 'test', 'cameras': scene.getTestCameras()}, 
        {'name': 'train', 'cameras': scene.getTrainCameras()}
    )

    for config in validation_configs:
        # 初始化指标计算器
        l1_test = 0
        psnr_test = 0
        ssim_test = 0
        lpips_test = 0
        
        # 遍历所有视角
        for viewpoint in config['cameras']:
            # 渲染和指标计算
            gt_image = viewpoint.original_image.cuda()
            image = torch.clamp(renderFunc(viewpoint, None, scene.gaussians, *renderArgs)["render"], 0.0, 1.0)
            
            # 累计各项指标
            l1_test += l1_loss(image, gt_image).mean()
            image = image[None, ...]
            gt_image = gt_image[None, ...]
            psnr_test += m_psnr(image, gt_image)
            ssim_test += m_ssim(image, gt_image)
            lpips_test += m_lpips(image, gt_image)

            # 保存测试图像
            if config['name'] == 'test':
                torchvision.utils.save_image(image, os.path.join(scene.model_path, "save_test", f"{viewpoint.image_name}.png"))
        
        # 计算平均指标
        psnr_test /= len(config['cameras'])
        ssim_test /= len(config['cameras'])
        lpips_test /= len(config['cameras'])
        l1_test /= len(config['cameras'])  
        
        # 打印和保存结果
        print(f"\n[ITER {iteration}] Evaluating {config['name']}: L1 {format(l1_test, '.4f')} "
              f"PSNR {format(psnr_test, '.4f')} SSIM {format(ssim_test, '.4f')} Lpips {format(lpips_test, '.4f')}")
        
        results[config['name']][iteration] = {
            'psnr': psnr_test.item(),
            'ssim': ssim_test.item(),
            'lpips': lpips_test.item(),
            'l1': l1_test.item()
        }

    # 保存结果到JSON文件
    torch.cuda.empty_cache()
    with open(os.path.join(scene.model_path, 'results.json'), 'w') as wf:
        json.dump(results, wf, indent=4)

def main():
    """主入口函数"""
    # 配置参数解析
    parser = ArgumentParser(description="Training script parameters")
    parser.add_argument("--base_cfg", type=str, default="./configs/gs_base.yaml")
    parser.add_argument("--data_cfg", type=str, default="./configs/nusc.yaml")
    parser.add_argument("--source_path", type=str, default="")
    parser.add_argument("--model_path", type=str, default="")
    args = parser.parse_args()
    
    # 合并配置文件
    cfg = OmegaConf.merge(OmegaConf.load(args.base_cfg), OmegaConf.load(args.data_cfg))
    if len(args.source_path) > 0:
        cfg.source_path = args.source_path
    if len(args.model_path) > 0:
        cfg.model_path = args.model_path
        
    print("Optimizing " + args.model_path)
    training(cfg)
    print("\nTraining complete.")

if __name__ == "__main__":
    main()
