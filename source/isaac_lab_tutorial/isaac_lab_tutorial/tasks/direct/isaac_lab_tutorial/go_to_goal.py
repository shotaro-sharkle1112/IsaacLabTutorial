# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# TODO:状態数の定義、状態番号を返す関数の定義、リモの瞬間移動の細かい動作修正、ゴール場所の描画とゴールのランダム化、そのQテーブルを保存する機能
# TODO:
import argparse
import math
import random

import onnxruntime as ort
import numpy as np

POLICY_PATH = r"C:\Users\xr\Issac\IssacLab\Projects\IsaacLabTutorial\source\isaac_lab_tutorial\isaac_lab_tutorial\tasks\direct\isaac_lab_tutorial\policy.onnx"
OBS_SHAPE = (3,)     # 例: CartPole-v1 なら観測 4 次元。ご自身の環境に合わせて！
ACTION_DIM = 2

# ========= ONNX モデルの読み込み =========
session = ort.InferenceSession(POLICY_PATH, providers=["CPUExecutionProvider"])

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
import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg, ArticulationCfg, RigidObjectCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
import isaaclab.utils.math as math_utils
from isaac_lab_tutorial.robots.limo import LIMO_CFG


x = np.random.randn(3, *OBS_SHAPE).astype(np.float32)
pm, pls = session.run(None, {"obs": x})
print("ONNX policy shapes:", pm.shape, pls.shape)

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
         radius=0.2,
         height=0.5,
         rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
         collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=False),
         visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0), metallic=0.2, opacity=0.5),
      ),
      init_state=RigidObjectCfg.InitialStateCfg(pos=(10.0, 10.0, 0.15)),
   )

   Limo = LIMO_CFG.replace(
      prim_path="/World/envs/env_.*/Robot",
      )

def limo_at_goal(scene:InteractiveScene, dist_threshold:float, goal:torch.Tensor) -> bool:
    limo_state = scene["Limo"].data.root_state_w.clone()
    c = limo_state[:, :3]
    dist_err = torch.norm(c[0] - goal, p=2).item()
    if dist_err <= dist_threshold:
        return True
    else:
        return False
    
def get_observation(scene, command) -> dict:
    forwards = math_utils.quat_apply(scene["Limo"].data.root_link_quat_w, scene["Limo"].data.FORWARD_VEC_B)
    # obs = torch.hstack((self.velocity, self.commands))
    
    dot = torch.sum(forwards * command, dim=-1, keepdim=True)
    cross = torch.cross(forwards, command, dim=-1)[:,-1].reshape(-1,1)
    forward_speed = scene["Limo"].data.root_com_lin_vel_b[:,0].reshape(-1,1)
    
    obs = np.array([[dot.item(), cross.item(), forward_speed.item()]], dtype=np.float32)
    return obs

    
def run_simulator(sim: sim_utils.SimulationContext, scene: InteractiveScene):
    
    # sim and learning
    sim_dt = sim.get_physics_dt()
    sim_time = 0.0
    torque = torch.tensor([[0.0,0.0,0.0,0.0]])
    learning_state = 0
    env_state = 0
    steps = 0
    episodes = 0
    pm  = np.zeros((1, ACTION_DIM), dtype=np.float32)   # 平均ベクトル
    pls = np.zeros((1, ACTION_DIM), dtype=np.float32)
    goal = torch.tensor([0.0, 0.0, 0.0],device=scene.device)
    initial_distance = 2.0
    steps_threshold = 1000
    command_change = 100
    dist_threshold = 0.2
    obs = np.zeros((1, 3), dtype=np.float32)

    # goal consists of [target_pos(3), target_ang(1)]
   

    while simulation_app.is_running():

        if learning_state == 0: # reset
            torque = torch.tensor([[0.0,0.0,0.0,0.0]])
            goal_direction = random.uniform(-math.pi, math.pi)
            goal[0:2] = torch.tensor([
                initial_distance*math.cos(goal_direction), 
                initial_distance*math.sin(goal_direction)
                ],device=scene.device)
            # ゴールの配置
            root_goal_state = scene["Goal"].data.default_root_state.clone()
            root_goal_state[0, :3] = goal[:3]
            scene["Goal"].write_root_pose_to_sim(root_goal_state[:, :7])
            # reset the scene entities to their initial positions offset by the environment origins
            root_Limo_state = scene["Limo"].data.default_root_state.clone()
            root_Limo_state[:, :3] += scene.env_origins

            # copy the default root state to the sim for the limo's orientation and velocity
            scene["Limo"].write_root_pose_to_sim(root_Limo_state[:, :7])
            scene["Limo"].write_root_velocity_to_sim(root_Limo_state[:, 7:])

            # copy the default joint states to the sim
            joint_pos, joint_vel = (
                scene["Limo"].data.default_joint_pos.clone(),
                scene["Limo"].data.default_joint_vel.clone(),
            )
            scene["Limo"].write_joint_state_to_sim(joint_pos, joint_vel)

            # clear internal buffers
            scene.reset()

            # change state
            learning_state = 1

            print("[INFO]: Resetting Limo state...")

        elif learning_state == 1: # select action

            # TODO: 環境情報の取得から
            # get env information
            # commandの生成：limoからゴールへのベクトル
            if steps == 0:
                command = goal - scene["Limo"].data.root_state_w.clone()[0,:3]
                command[2] = 0.0
                command = command / (torch.norm(command) + 1e-8)
            print("[INFO]: command",command)
            obs = get_observation(scene, command)
            # select action
            pm, pls = session.run(None, {"obs": obs})
            # change state
            learning_state = 2

        elif learning_state == 2: # move
         # NOTE:ここは何も書かない
            # change state
            learning_state = 3

        elif learning_state == 3: # act
            left_torque = pm[0][0]
            right_torque = pm[0][1]
            if steps % 20 == 0:
                print(f"[INFO]: obs {obs}")
                print(f"[INFO]: left {left_torque}, right {right_torque}")
            
            torque = -10.0 * torch.tensor([[left_torque, right_torque,left_torque,right_torque]])
            # limoのトルクをかける
            scene["Limo"].set_joint_velocity_target(torque)
            steps += 1 
            learning_state = 4

        elif learning_state == 4: # calculate reward and update Q-table
            # NOTE:trainではないので必要ない

            # go back to selecting action
            learning_state = 1
            
        # check the termination conditions
        # box is goal location or steps over
        if limo_at_goal(scene, dist_threshold, goal) or steps == steps_threshold:
            learning_state = 0
            steps = 0
            episodes += 1
            print(f"[INFO]: episodes {episodes}")

        

        
        sim.step()
        scene.write_data_to_sim()
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