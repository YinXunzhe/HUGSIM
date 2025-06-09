import hugsim_env
import numpy as np
from sim.utils.score_calculator import hugsim_evaluate
import open3d as o3d
from omegaconf import OmegaConf
from sim.utils.launch_ad import launch, check_alive
import json
import pickle
from sim.utils.sim_utils import traj2control, traj_transform_to_global
from argparse import ArgumentParser
import gymnasium
import sys
import os
os.environ["SDL_AUDIODRIVER"] = "dummy"  # 禁用音频驱动
sys.path.append(os.getcwd())
sys.path.append(os.getcwd() + "/sim")


try:
    import pygame
    from pygame.locals import *
except ImportError:
    raise RuntimeError(
        'cannot import pygame, make sure pygame package is installed')


class KeyboardDriver(object):
    """Class that handles keyboard input."""

    def __init__(self):
        self._ego = None
        self.acc = 0
        self.steer_rate = 0


    def parse_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return True

        keys = pygame.key.get_pressed()

        # ===== 加速度控制 =====
        target_acc = 0.0
        if keys[K_UP]:
            target_acc = 2.0
        elif keys[K_DOWN]:
            target_acc = -2.0

        # 平滑逼近目标加速度
        acc_alpha = 0.1  # 越小越慢
        self.acc += acc_alpha * (target_acc - self.acc)

        # 松开时逐渐减速（模拟滑行）
        if not keys[K_UP] and not keys[K_DOWN]:
            self.acc *= 0.95  # 模拟滑行阻力

        # ===== 转向控制 =====
        target_steer_rate = 0.0
        if keys[K_LEFT]:
            target_steer_rate = 0.8
        elif keys[K_RIGHT]:
            target_steer_rate = -0.8

        # 平滑逼近目标转向速率
        steer_alpha = 0.15  # 越小越平滑
        self.steer_rate += steer_alpha * (target_steer_rate - self.steer_rate)

        # 松开时转向回正
        if not keys[K_LEFT] and not keys[K_RIGHT]:
            self.steer_rate *= 0.85  # 模拟回正力矩

        # ===== 紧急停止 =====
        if keys[K_SPACE]:
            self.acc = 0.0
            self.steer_rate = 0.0

        return False


class Player(object):
    def __init__(self):
        pygame.init()
        pygame.font.init()
        self.display = pygame.display.set_mode(
            (2400, 1000),
            pygame.HWSURFACE | pygame.DOUBLEBUF)
        self.display.fill((0, 0, 0))
        pygame.display.flip()
        self.font = pygame.font.Font(pygame.font.get_default_font(), 20)

    def render(self, info, obs, driver: KeyboardDriver):
        self.display.fill((0, 0, 0))

        u_offset = 0
        surface = pygame.surfarray.make_surface(
            obs['rgb']['CAM_FRONT_LEFT'].swapaxes(0, 1))
        self.display.blit(surface, (u_offset, 0))
        surface = pygame.surfarray.make_surface(
            obs['rgb']['CAM_BACK_LEFT'].swapaxes(0, 1))
        self.display.blit(surface, (u_offset, 550))
        u_offset += 800
        surface = pygame.surfarray.make_surface(
            obs['rgb']['CAM_FRONT'].swapaxes(0, 1))
        self.display.blit(surface, (u_offset, 0))
        surface = pygame.surfarray.make_surface(
            obs['rgb']['CAM_BACK'].swapaxes(0, 1))
        self.display.blit(surface, (u_offset, 550))
        u_offset += 800
        surface = pygame.surfarray.make_surface(
            obs['rgb']['CAM_FRONT_RIGHT'].swapaxes(0, 1))
        self.display.blit(surface, (u_offset, 0))
        surface = pygame.surfarray.make_surface(
            obs['rgb']['CAM_BACK_RIGHT'].swapaxes(0, 1))
        self.display.blit(surface, (u_offset, 550))

        color = (255, 255, 255)
        v_offset = 450
        surface = self.font.render(
            f'driver-------acc:{driver.acc:.1f}, steer_rate:{driver.steer_rate:.1f}', True, color)
        self.display.blit(surface, (0, v_offset))
        v_offset += 18
        surface = self.font.render(
            f'time_stamp:{info["timestamp"]}, collision:{info["collision"]}', True, color)
        self.display.blit(surface, (0, v_offset))
        v_offset += 18
        surface = self.font.render(f'ego pose:{info["ego_pos"]}', True, color)
        self.display.blit(surface, (0, v_offset))
        v_offset += 18
        surface = self.font.render(f'ego rot:{info["ego_rot"]}', True, color)
        self.display.blit(surface, (0, v_offset))
        v_offset += 18
        surface = self.font.render(
            f'velo:{info["ego_velo"]:.1f}, accelerate:{info["accelerate"]:.1f}, steer_rate:{info["steer_rate"]:.1f}', True, color)
        self.display.blit(surface, (0, v_offset))
        # v_offset += 18
        pygame.display.flip()


def game_loop(cfg, output):

    env = gymnasium.make('hugsim_env/HUGSim-v0', cfg=cfg, output=output)

    obs, info = env.reset()
    done = False
    cnt = 0
    save_data = {'type': 'closeloop', 'frames': []}

    obs_pipe = os.path.join(output, 'obs_pipe')
    if not os.path.exists(obs_pipe):
        os.mkfifo(obs_pipe)
    # plan_pipe = os.path.join(output, 'plan_pipe')
    # if not os.path.exists(plan_pipe):
    #     os.mkfifo(plan_pipe)
    print('Ready for simulation')

    driver = KeyboardDriver()

    player = Player()

    obs, info = None, None
    while not done:

        if obs is None or info is None:
            obs, info = env.reset()

        print('ego pose', info['ego_pos'])

        done = driver.parse_events()
        action = {'acc': driver.acc, 'steer_rate': -driver.steer_rate}
        obs, reward, terminated, truncated, info = env.step(action)
        cnt += 1
        # done = terminated or truncated or cnt > 400

        # save_data['frames'].append({
        #     'time_stamp': info['timestamp'],
        #     'is_key_frame': True,
        #     'ego_box': info['ego_box'],
        #     'obj_boxes': info['obj_boxes'],
        #     'obj_names': ['car' for _ in info['obj_boxes']],
        #     'planned_traj': {
        #         'traj': [],
        #         'timestep': 0.5
        #     },
        #     'collision': info['collision'],
        #     'rc': info['rc']
        # })

        player.render(info, obs, driver)

    with open(obs_pipe, "wb") as pipe:
        pipe.write(pickle.dumps('Done'))

    with open(os.path.join(output, 'data.pkl'), 'wb') as wf:
        pickle.dump([save_data], wf)

    ground_xyz = np.asarray(o3d.io.read_point_cloud(
        os.path.join(output, 'ground.ply')).points)
    scene_xyz = np.asarray(o3d.io.read_point_cloud(
        os.path.join(output, 'scene.ply')).points)
    results = hugsim_evaluate([save_data], ground_xyz, scene_xyz)
    with open(os.path.join(output, 'eval.json'), 'w') as f:
        json.dump(results, f)


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    parser.add_argument("--scenario_path", type=str, required=True)
    parser.add_argument("--base_path", type=str, required=True)
    parser.add_argument("--camera_path", type=str, required=True)
    parser.add_argument("--kinematic_path", type=str, required=True)
    args = parser.parse_args()

    scenario_config = OmegaConf.load(args.scenario_path)
    base_config = OmegaConf.load(args.base_path)
    camera_config = OmegaConf.load(args.camera_path)
    kinematic_config = OmegaConf.load(args.kinematic_path)
    cfg = OmegaConf.merge(
        {"scenario": scenario_config},
        {"base": base_config},
        {"camera": camera_config},
        {"kinematic": kinematic_config}
    )
    cfg.base.output_dir = cfg.base.output_dir + "/manual"

    model_path = os.path.join(
        cfg.base.model_base, cfg.scenario.scene_name,"exported")
    model_config = OmegaConf.load(os.path.join(model_path, 'cfg.yaml'))
    cfg.update(model_config)

    output = os.path.join(cfg.base.output_dir,
                          cfg.scenario.scene_name+"_"+cfg.scenario.mode)
    os.makedirs(output, exist_ok=True)

    try:
        game_loop(cfg, output)
    except Exception as e:
        print(e)

    # # For debug
    # create_gym_env(cfg, output)
