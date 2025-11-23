# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from isaac_lab_tutorial.robots.jetbot import JETBOT_CONFIG
from isaac_lab_tutorial.robots.limo import LIMO_FRONT_CFG, LIMO_CFG

from isaaclab.assets import ArticulationCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass



@configclass
class IsaacLabTutorialEnvCfg(DirectRLEnvCfg):
    # env
    decimation = 2
    episode_length_s = 5.0
    # - spaces definition
    action_space = 2
    # observation_space = 9
    observation_space = 3
    state_space = 0
    # simulation
    sim: SimulationCfg = SimulationCfg(dt=1 / 120, render_interval=decimation)
    # robot(s)
    robot_cfg: ArticulationCfg = JETBOT_CONFIG.replace(prim_path="/World/envs/env_.*/Robot")
    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=100, env_spacing=2.0, replicate_physics=True)
    dof_names = ["left_wheel_joint", "right_wheel_joint"]

@configclass
class LimoEnvCfg(DirectRLEnvCfg):
    # env
    decimation = 2
    episode_length_s = 5.0
    # - spaces definition
    action_space = 2
    # observation_space = 9
    observation_space = 3
    state_space = 0
    # simulation
    sim: SimulationCfg = SimulationCfg(dt=1 / 120, render_interval=decimation)
    # robot(s)
    robot_cfg: ArticulationCfg = LIMO_CFG.replace(prim_path="/World/envs/env_.*/Robot")
    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=100, env_spacing=2.0, replicate_physics=True)
    dof_names = ["front_left_wheel","front_right_wheel", "rear_left_wheel", "rear_right_wheel"]

@configclass
class LimoPendulumEnvCfg(DirectRLEnvCfg):
    # env
    decimation = 2
    episode_length_s = 30.0
    action_scale = 30.0
    # - spaces definition
    action_space = 1
    # observation_space = 9
    observation_space = 4
    state_space = 0
    # simulation
    sim: SimulationCfg = SimulationCfg(dt=1 / 120, render_interval=decimation)
    # robot(s)
    robot_cfg: ArticulationCfg = LIMO_CFG.replace(prim_path="/World/envs/env_.*/Robot")
    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=100, env_spacing=4.0, replicate_physics=True)
    cart_dof_names = ["front_left_wheel","front_right_wheel", "rear_left_wheel", "rear_right_wheel"]
    pole_dof_name = ["pendulum"]

    max_cart_pos = 10000.0  # the cart is reset if it exceeds that position [m]
    initial_pole_angle_range = [-0.1, 0.1]  # the range in which the pole angle is sampled from on reset [rad]
    weight_range = [0.1, 0.3]
    # reward scales
    rew_scale_alive = 1.0
    rew_scale_terminated = -2.0
    rew_scale_pole_pos = -1.0
    rew_scale_pole_vel = -0.01
    rew_scale_cart_vel = -0.001
    rew_scale_cart_pos = -0.001
    

@configclass
class LimoPendulumEnvCfgC10s(DirectRLEnvCfg):
    # env
    decimation = 2
    episode_length_s = 10.0
    action_scale = 100.0
    # - spaces definition
    action_space = 1
    # observation_space = 9
    observation_space = 4
    state_space = 0
    # simulation
    sim: SimulationCfg = SimulationCfg(dt=1 / 120, render_interval=decimation)
    # robot(s)
    robot_cfg: ArticulationCfg = LIMO_CFG.replace(prim_path="/World/envs/env_.*/Robot")
    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=100, env_spacing=4.0, replicate_physics=True)
    cart_dof_names = ["front_left_wheel","front_right_wheel", "rear_left_wheel", "rear_right_wheel"]
    pole_dof_name = ["pendulum"]

    max_cart_pos = 3.0  # the cart is reset if it exceeds that position [m]
    initial_pole_angle_range = [-0.1, 0.1]  # the range in which the pole angle is sampled from on reset [rad]
    weight_range = [0.1, 0.3]
    # reward scales
    rew_scale_alive = 1.0
    rew_scale_terminated = -2.0
    rew_scale_pole_pos = -1.0
    rew_scale_pole_vel = -0.005
    rew_scale_cart_vel = -0.01
    rew_scale_cart_pos = -0.8

@configclass
class LimoPendulumEnvCfgC60s(DirectRLEnvCfg):
    # env
    decimation = 2
    episode_length_s = 60.0
    action_scale = 100.0
    # - spaces definition
    action_space = 1
    # observation_space = 9
    observation_space = 4
    state_space = 0
    # simulation
    sim: SimulationCfg = SimulationCfg(dt=1 / 120, render_interval=decimation)
    # robot(s)
    robot_cfg: ArticulationCfg = LIMO_CFG.replace(prim_path="/World/envs/env_.*/Robot")
    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=100, env_spacing=4.0, replicate_physics=True)
    cart_dof_names = ["front_left_wheel","front_right_wheel", "rear_left_wheel", "rear_right_wheel"]
    pole_dof_name = ["pendulum"]

    max_cart_pos = 3.0  # the cart is reset if it exceeds that position [m]
    initial_pole_angle_range = [-0.1, 0.1]  # the range in which the pole angle is sampled from on reset [rad]
    weight_range = [0.1, 0.3]
    # reward scales
    rew_scale_alive = 1.0
    rew_scale_terminated = -2.0
    rew_scale_pole_pos = -1.0
    rew_scale_pole_vel = -0.005
    rew_scale_cart_vel = -0.01
    rew_scale_cart_pos = -0.8