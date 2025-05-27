import pygame
import numpy as np
import torch
from pygame.locals import *
from omegaconf import OmegaConf
import os
import pickle
import open3d as o3d
from gaussian_renderer import render
from scene.gaussian_model import GaussianModel
from sim.utils.sim_utils import create_cam, rt2pose, pose2rt, load_camera_cfg
from scipy.spatial.transform import Rotation as SCR


class HugSimViewer:
    def __init__(self, model_path, camera_path, iteration=7000):
        pygame.init()
        pygame.font.init()

        # 初始化显示窗口
        self.display = pygame.display.set_mode((2400, 1000),
                                               pygame.HWSURFACE | pygame.DOUBLEBUF)
        self.display.fill((0, 0, 0))
        pygame.display.flip()
        self.font = pygame.font.Font(pygame.font.get_default_font(), 20)

        # 加载模型
        self.gaussians = GaussianModel(3)  # sh_degree=3
        (model_params, _) = torch.load(
            os.path.join(model_path, "ckpts", f"chkpnt{iteration}.pth"),
            weights_only=False)
        self.gaussians.restore(model_params, None)

        # 加载相机配置
        self.cam_params, _, self.cam_rect = load_camera_cfg(
            OmegaConf.load(camera_path))

        # 相机控制参数
        self.camera_pos = np.array([0.0, 0.0, 0.0])  # 相机位置
        self.camera_yaw = 0  # 偏航角
        self.camera_pitch = 0  # 俯仰角
        self.move_speed = 0.1
        self.rotate_speed = 0.01

        # 渲染参数
        self.bg_color = torch.tensor(
            [0, 0, 0], dtype=torch.float32, device="cuda")
        self.render_kwargs = {
            "pc": self.gaussians,
            "bg_color": self.bg_color,
            "dynamic_gaussians": {},
            "unicycles": None,
            "planning": [{}, {}]
        }

    def handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False

        # 键盘控制
        keys = pygame.key.get_pressed()

        move_speed_fac = 2
        # 前后移动 (z轴方向)
        if keys[K_w] or keys[K_s]:
            move_dir = np.array(
                [-np.sin(self.camera_yaw), 0, np.cos(self.camera_yaw)])
            if keys[K_w]:
                self.camera_pos += move_dir * self.move_speed*move_speed_fac
            else:
                self.camera_pos -= move_dir * self.move_speed*move_speed_fac

        # 左右移动 (x轴方向)
        if keys[K_a] or keys[K_d]:
            strafe_dir = np.array(
                [np.cos(self.camera_yaw), 0, np.sin(self.camera_yaw)])
            if keys[K_a]:
                self.camera_pos -= strafe_dir * self.move_speed
            else:
                self.camera_pos += strafe_dir * self.move_speed

        # 上下移动
        if keys[K_q] or keys[K_e]:
            if keys[K_q]:
                self.camera_pos[1] -= self.move_speed
            else:
                self.camera_pos[1] += self.move_speed

        # 旋转控制 (符合右手法则)
        if keys[K_LEFT]:
            self.camera_yaw -= self.rotate_speed  # 左转：绕Y轴负旋转
        if keys[K_RIGHT]:
            self.camera_yaw += self.rotate_speed  # 右转：绕Y轴正旋转
        if keys[K_UP]:
            self.camera_pitch = np.clip(
                self.camera_pitch + self.rotate_speed, -np.pi/2, np.pi/2)
        if keys[K_DOWN]:
            self.camera_pitch = np.clip(
                self.camera_pitch - self.rotate_speed, -np.pi/2, np.pi/2)

        return True

    def render_views(self):
        self.display.fill((0, 0, 0))

        # 获取当前相机姿态 (yaw绕Y轴，pitch绕X轴，roll绕Z轴)
        # 使用'yxz'欧拉角顺序: yaw -> pitch -> roll(这里roll=0)
        rot = SCR.from_euler(
            'yxz', [self.camera_yaw, self.camera_pitch, 0]).as_matrix()
        c2w = np.eye(4)
        c2w[:3, :3] = rot
        c2w[:3, 3] = self.camera_pos

        # # 调试输出
        # print(
        #     f"Camera pos: {self.camera_pos}, yaw: {np.degrees(self.camera_yaw):.1f}°, pitch: {np.degrees(self.camera_pitch):.1f}°")

        # 渲染各个视角
        u_offset = 0
        for cam_name in ['CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT']:
            # 计算相机变换矩阵
            v2c = self.cam_params[cam_name]["v2c"]
            c2front = self.cam_params["CAM_FRONT"]["v2c"] @ np.linalg.inv(
                v2c) @ self.cam_rect
            cam_c2w = c2w @ c2front
            # print(f"{cam_name} c2w:\n{cam_c2w}")  # 调试输出变换矩阵

            # 创建相机并渲染
            viewpoint = create_cam(
                self.cam_params[cam_name]["intrinsic"], cam_c2w)
            render_pkg = render(viewpoint=viewpoint,
                                prev_viewpoint=None, **self.render_kwargs)

            # 转换并显示图像
            rgb = (torch.permute(render_pkg['render'].clamp(
                0, 1), (1, 2, 0)).detach().cpu().numpy() * 255).astype(np.uint8)
            surface = pygame.surfarray.make_surface(rgb.swapaxes(0, 1))
            self.display.blit(surface, (u_offset, 0))

            u_offset += 800

        # # 渲染后视角
        # u_offset = 0
        # for cam_name in ['CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT']:
        #     v2c = self.cam_params[cam_name]["v2c"]
        #     c2front = self.cam_params["CAM_FRONT"]["v2c"] @ np.linalg.inv(
        #         v2c) @ self.cam_rect
        #     cam_c2w = c2w @ c2front

        #     viewpoint = create_cam(
        #         self.cam_params[cam_name]["intrinsic"], cam_c2w)
        #     render_pkg = render(viewpoint=viewpoint,
        #                         prev_viewpoint=None, **self.render_kwargs)

        #     rgb = (torch.permute(render_pkg['render'].clamp(
        #         0, 1), (1, 2, 0)).detach().cpu().numpy() * 255).astype(np.uint8)
        #     surface = pygame.surfarray.make_surface(rgb.swapaxes(0, 1))
        #     self.display.blit(surface, (u_offset, 550))

        #     u_offset += 800

        # 显示相机位置信息
        color = (255, 255, 255)
        v_offset = 450
        info_text = [
            f'Camera Position: {self.camera_pos}',
            f'Camera Yaw: {self.camera_yaw:.2f}',
            f'Camera Pitch: {self.camera_pitch:.2f}',
            'Controls: WASD - Move, Q/E - Up/Down, Arrows - Rotate'
        ]

        for text in info_text:
            surface = self.font.render(text, True, color)
            self.display.blit(surface, (0, v_offset))
            v_offset += 18

        pygame.display.flip()

    def run(self):
        running = True
        while running:
            running = self.handle_events()
            self.render_views()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to the trained model")
    parser.add_argument("--camera_path", type=str, default="./configs/sim/zone_camera.yaml",
                        help="Path to camera configuration file")
    parser.add_argument("--iteration", type=int, default=30000,
                        help="Iteration number of the model to load")
    args = parser.parse_args()

    viewer = HugSimViewer(args.model_path, args.camera_path, args.iteration)
    viewer.run()
