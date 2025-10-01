# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# TODO:状態数の定義、状態番号を返す関数の定義、リモの瞬間移動の細かい動作修正、ゴール場所の描画とゴールのランダム化、そのQテーブルを保存する機能
# TODO:
import argparse
import math

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(
    description="This script demonstrates adding a custom robot to an Isaac Lab environment."
)
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to spawn.")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

from dataclasses import dataclass
import numpy as np
import torch
import random

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg, ArticulationCfg, RigidObjectCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg

from SQKF.assets.limo import LIMO_CFG
from SQKF.algorithms.sqkf_test import SQKF, SQKFCfg

def quat_from_yaw_wxyz(yaw_rad: torch.Tensor) -> torch.Tensor:
    """
    yaw(ラジアン) -> クォータニオン(w,x,y,z)
    ・Z軸回りの回転のみ（roll=pitch=0）
    ・入力はスカラー or 任意形状の tensor。出力は (..., 4)
    """
    # 0.5 * yaw を使う
    half = 0.5 * yaw_rad
    c = torch.cos(half)
    s = torch.sin(half)
    # roll=pitch=0 のときの式（右手系）
    # w = cos(yaw/2), x = 0, y = 0, z = sin(yaw/2)
    w = c
    x = torch.zeros_like(c)
    y = torch.zeros_like(c)
    z = s
    q = torch.stack([w, x, y, z], dim=-1)
    # 念のため正規化
    q = q / torch.linalg.vector_norm(q, dim=-1, keepdim=True).clamp(min=1e-12)
    return q

def quat_from_yaw_deg_wxyz(yaw_deg: torch.Tensor) -> torch.Tensor:
    """
    yaw(度) -> クォータニオン(w,x,y,z)
    """
    return quat_from_yaw_wxyz(torch.deg2rad(yaw_deg))

# 環境の定義はこちらに
@dataclass
class EnvCfg:

    # patterns
    dist_patterns_num:int = 6
    ang_patterns_num:int = 6

    # 

    # Box (width > depth)
    box_width:float = 1.0
    box_depth:float = 0.5
    box_hight:float = 0.1

    box_push_point = 6 # this means action_num

    box_ang_push:float = 30.0

    box_push_pos_radius_03:float = box_width/2.0 + 0.2
    box_push_pos_radius_1245:float = box_depth/2.0 + 0.3

    # initial distance between goal and box
    initial_distance:float = 2.0
    # 離散化される最大の距離、これ以上距離が離れると大きな罰則を与える
    max_distance:float = 2.2

    # 一応用意はしてあるが固定値である
    max_angle:float = 360.0

    # reward
    rew_pos_scale:float = 1.0
    rew_ang_scale:float = 1.0
    rew_time:float = 1.0
    rew_pos_penalty:float = 1000.0
    rew_ang_penalty:float = 1000.0

    def compute_rewards(self, scene:InteractiveScene, goal:torch.Tensor) -> float:
        box_state = scene["Box"].data.root_state_w.clone()
        dist_err = torch.norm(box_state[0, :3] - goal[:3]).item()

        q = box_state[:, 3:7]
        w, x, y, z = q.unbind(-1)

        # 正規化（安全のため）
        norm = torch.clamp(torch.sqrt(x*x + y*y + z*z + w*w), min=1e-8)
        x, y, z, w = x/norm, y/norm, z/norm, w/norm

        yaw = torch.atan2(2*(w*z + x*y), 1 - 2*(y*y + z*z))
        ang_err = 0.0
        box_ang = 0.0
        if torch.rad2deg(yaw).item() <= 0.0:
            box_ang = torch.pi + torch.rad2deg(yaw).item()
        else:
            box_ang = torch.rad2deg(yaw).item()
        ang_err = goal[3].item() - box_ang

        ang_err = math.fabs(ang_err)

        print("[INFO]: ang_err", ang_err)
        print("[INFO]: ang_box", torch.rad2deg(yaw).item())
        print("[INFO]: ang_goal", goal[3])
        total_reward = -self.rew_pos_scale*dist_err - self.rew_ang_scale*ang_err - self.rew_time
        if dist_err >= self.max_distance:
            total_reward -=  self.rew_ang_penalty
        return total_reward

env_cfg = EnvCfg()

def discretize(continuous, max_distance, num_bins):
    """
    指定した最大距離とビン数に基づき、距離を離散化する関数
    infは使わず、max_distanceを超える値は最後のビンに入れる

    Parameters
    ----------
    distance : float
        離散化したい単一の距離
    max_distance : float
        最大距離（これを超える値は最後のビンに入れる）
    num_bins : int
        離散化のビン数

    Returns
    -------
    label : int
        距離に対応する離散化されたラベル（0, 1, 2, ...）
    """

    # ビン境界（例: [0, 20, 40, 60, 80, 100]）
    bins = np.linspace(0, max_distance, num_bins + 1)

    # 最大距離を超えたら最後のビンに割り当てる
    if continuous >= max_distance:
        label = num_bins - 1
    else:
        label = np.digitize(continuous, bins, right=False) - 1

    return label

# TODO:なんで状態が29で固定になっちゃってるのかを調査しないといけない
# state_num = dist_err_num * ang_patterns_num + ang_err_num
def getEnvStateNum(env_cfg:EnvCfg, scene:InteractiveScene, goal:torch.Tensor) -> int:
    box_state = scene["Box"].data.root_state_w.clone()
    c = box_state[:, :3]
    dist_err = torch.norm(c[0] - goal[:3], p=2)

    q = box_state[:, 3:7]
    w, x, y, z = q.unbind(-1)

    # 正規化（安全のため）
    norm = torch.clamp(torch.sqrt(x*x + y*y + z*z + w*w), min=1e-8)
    x, y, z, w = x/norm, y/norm, z/norm, w/norm

    yaw = torch.atan2(2*(w*z + x*y), 1 - 2*(y*y + z*z))
    yaw = torch.as_tensor(yaw, device=scene.device)

    ang_err = 0.0
    box_ang = 0.0
    if torch.rad2deg(yaw).item() <= 0.0:
        box_ang = torch.pi + torch.rad2deg(yaw).item()
    else:
        box_ang = torch.rad2deg(yaw).item()
    ang_err = goal[3].item() - box_ang

    # TODO:学習を何回も重ねることでパターンを変える必要あり
    dist_pattern = discretize(dist_err.item(), env_cfg.max_distance, env_cfg.dist_patterns_num)
    ang_pattern = discretize(float(ang_err), env_cfg.max_angle, env_cfg.ang_patterns_num)

    state_num = dist_pattern*env_cfg.ang_patterns_num + ang_pattern
    return state_num

# TODO: それぞれのinteractivesceneにデータを格納できるかを調べる
class LimoSceneCfg(InteractiveSceneCfg):
    """Designs the scene."""

    # Ground-plane
    ground = AssetBaseCfg(prim_path="/World/defaultGroundPlane", spawn=sim_utils.GroundPlaneCfg())

    # lights
    dome_light = AssetBaseCfg(
        prim_path="/World/Light", spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
    )

    Goal = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Goal",
        spawn=sim_utils.ConeCfg(
            size=(env_cfg.box_width, env_cfg.box_depth, env_cfg.box_hight),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=False),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0), metallic=0.2, opacity=0.5),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(10.0, 10.0, 0.15)),
    )

   
    


    
def run_simulator(sim: sim_utils.SimulationContext, scene: InteractiveScene):

    
    # sim and learning
    sim_dt = sim.get_physics_dt()
    sim_time = 0.0
    count = 0
    torque = torch.tensor([[0.0,0.0,0.0,0.0]])
    learning_state = 0
    env_state = 0
    steps = 0
    episodes = 0
    location = 0
    # goal consists of [target_pos(3), target_ang(1)]
    goal = torch.zeros(4, device=scene.device)
    reward = 0.0
    agent_num = 2
    action_num = 6
    env_num = env_cfg.ang_patterns_num*env_cfg.dist_patterns_num

    # learning cfg is here
    episodes_threshold = 1500
    action_threshold = 20
    steps_threshold = 100
    dist_threshold = 0.5
    ang_threshold = 40.0
    
    # TODO:状態数(箱とゴールの距離、角度)の定義と、距離、角度を代入することで離散化してくれる関数を作成する
    sqkfcfg = SQKFCfg(num_robots=agent_num, num_actions=action_num, num_states=env_num)
    sqkf = SQKF(sqkfcfg)

    while simulation_app.is_running():

        if learning_state == 0: # reset
            # TODO:ゴールの位置の初期化
            goal[3] = random.uniform(0.0, 180.0)
            goal_direction = random.uniform(-math.pi, math.pi)
            goal[0:2] = torch.tensor([
                env_cfg.initial_distance*math.cos(goal_direction), 
                env_cfg.initial_distance*math.sin(goal_direction)
                ],device=scene.device)
            # ゴールの配置
            root_goal_state = scene["Goal"].data.default_root_state.clone()
            root_goal_state[0, :3] = goal[:3]
            root_goal_state[0, 3:7] = quat_from_yaw_deg_wxyz(goal[3])
            scene["Goal"].write_root_pose_to_sim(root_goal_state[:, :7])
            env_state = getEnvStateNum(env_cfg, scene, goal)
            # reset counters
            count = 0
            torque = torch.tensor([[0.0, 0.0, 0.0, 0.0]])
            # reset the scene entities to their initial positions offset by the environment origins
            root_limo0_state = scene["Limo0"].data.default_root_state.clone()
            root_limo0_state[:, :3] += scene.env_origins
            root_limo1_state = scene["Limo1"].data.default_root_state.clone()
            root_limo1_state[:, :3] += scene.env_origins

            root_box_state = scene["Box"].data.default_root_state.clone()
            root_box_state[:, :3] += scene.env_origins

            # copy the default root state to the sim for the jetbot's orientation and velocity
            scene["Limo0"].write_root_pose_to_sim(root_limo0_state[:, :7])
            scene["Limo0"].write_root_velocity_to_sim(root_limo0_state[:, 7:])
            scene["Limo1"].write_root_pose_to_sim(root_limo1_state[:, :7])
            scene["Limo1"].write_root_velocity_to_sim(root_limo1_state[:, 7:])

            scene["Box"].write_root_pose_to_sim(root_box_state[:, :7])

            # copy the default joint states to the sim
            joint_pos, joint_vel = (
                scene["Limo0"].data.default_joint_pos.clone(),
                scene["Limo0"].data.default_joint_vel.clone(),
            )
            scene["Limo0"].write_joint_state_to_sim(joint_pos, joint_vel)
            joint_pos, joint_vel = (
                scene["Limo1"].data.default_joint_pos.clone(),
                scene["Limo1"].data.default_joint_vel.clone(),
            )
            scene["Limo1"].write_joint_state_to_sim(joint_pos, joint_vel)

            # clear internal buffers
            scene.reset()

            # change state
            learning_state = 1

            print("[INFO]: Resetting Limo state...")

        elif learning_state == 1: # select action
            # TODO: sqkf.pyの関数呼び出し
            print("[INFO]: Select a action...")
            torque = torch.tensor([[0.0,0.0,0.0,0.0]])

            # get env information
            env_state = getEnvStateNum(env_cfg, scene, goal)
            # select action
            actions = sqkf.select_actions(env_state)
            # change state
            learning_state = 2

        elif learning_state == 2: # move
            print("[INFO]: Move Limo...")
            torque = torch.tensor([[0.0,0.0,0.0,0.0]])
            limo_target_state = getLimoPos(scene, actions[0])

            # depending on the action, move limo to side of box
            limo_target_state[:, :3] += scene.env_origins
            scene["Limo0"].write_root_pose_to_sim(limo_target_state[:, :7])

            limo_target_state = getLimoPos(scene, actions[1])
            # depending on the action, move limo to side of box
            limo_target_state[:, :3] += scene.env_origins
            scene["Limo1"].write_root_pose_to_sim(limo_target_state[:, :7])

            # change state
            learning_state = 3

        elif learning_state == 3: # act
            if count == 0:
                print("[INFO]: Limo act...")
            torque = torch.tensor([[10.0, 10.0, 10.0, 10.0]])
            count += 1
            if count == action_threshold:
                # reset count
                count = 0
                # change state
                learning_state = 4

        elif learning_state == 4: # calculate reward and update Q-table
            print("[INFO]: calculate reward and update Q-table...")
            # TODO: sqkfの関数呼び出しによりQテーブルの更新を行う
            # calculate reward
            print("[INFO]: ",env_state)
            reward = env_cfg.compute_rewards(scene, goal)
            print("[INFO]: global reward", reward)
            # estimate each reward
            rewards = sqkf.estimate(env_state, reward, steps)
            print("[INFO]: each reward", rewards)

            sqkf.update(env_state, getEnvStateNum(env_cfg,scene,goal), rewards, actions)

            box_state = scene["Box"].data.root_state_w.clone()
            print("[INFO]: distance",torch.norm(box_state[0, :3] - goal[:3]).item())
            # increment steps
            steps += 1
            
            print(f"[INFO]: steps {steps}/{steps_threshold}")
            print(f"[INFO]: episodes {episodes}/{episodes_threshold}")
            print()

            # go back to selecting action
            learning_state = 1
            
        # check the termination conditions
        # box is goal location or steps over
        if box_at_goal(scene, dist_threshold, ang_threshold, goal) or steps == steps_threshold:
            learning_state = 0
            steps = 0
            episodes += 1
            sqkf.save_q("runs/exp2/q_tables.npz")
            print(f"[INFO]: episodes {episodes}/{episodes_threshold}")
        
        if episodes == episodes_threshold:
            print("[INFO]: Learning finished")
            # learning finished
            return

        # limoのトルクをかける
        scene["Limo0"].set_joint_velocity_target(torque)
        scene["Limo1"].set_joint_velocity_target(torque)

        scene.write_data_to_sim()
        sim.step()
        sim_time += sim_dt
        scene.update(sim_dt)





def main():
    """Main function."""
    # Initialize the simulation context
    sim_cfg = sim_utils.SimulationCfg(device=args_cli.device)
    sim = sim_utils.SimulationContext(sim_cfg)
    sim.set_camera_view([3.5, 0.0, 3.2], [0.0, 0.0, 0.5])
    # Design scene
    scene_cfg = LimoSceneCfg(args_cli.num_envs, env_spacing=2.0)
    scene = InteractiveScene(scene_cfg)
    # Play the simulator
    sim.reset()
    # Now we are ready!
    print("[INFO]: Setup complete...")
    # Run the simulator
    run_simulator(sim, scene)


if __name__ == "__main__":
    main()
    simulation_app.close()