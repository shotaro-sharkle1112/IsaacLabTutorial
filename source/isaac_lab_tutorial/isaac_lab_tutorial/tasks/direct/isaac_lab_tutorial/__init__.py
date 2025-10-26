# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import gymnasium as gym

from . import agents

##
# Register Gym environments.
##


gym.register(
    id="Template-Isaac-Lab-Tutorial-Direct-v0",
    entry_point=f"{__name__}.isaac_lab_tutorial_env:IsaacLabTutorialEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.isaac_lab_tutorial_env_cfg:IsaacLabTutorialEnvCfg",
        "skrl_amp_cfg_entry_point": f"{agents.__name__}:skrl_amp_cfg.yaml",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:PPORunnerCfg",
    },
)

gym.register(
    id="Template-Isaac-Lab-Tutorial-Direct-Noise01",
    entry_point=f"{__name__}.isaac_lab_tutorial_env:IsaacLabTutorialEnvWithNoise01",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.isaac_lab_tutorial_env_cfg:IsaacLabTutorialEnvCfg",
        "skrl_amp_cfg_entry_point": f"{agents.__name__}:skrl_amp_cfg.yaml",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:PPORunnerCfg",
    },
)

gym.register(
    id="Template-Isaac-Lab-Tutorial-Direct-Noise02",
    entry_point=f"{__name__}.isaac_lab_tutorial_env:IsaacLabTutorialEnvWithNoise02",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.isaac_lab_tutorial_env_cfg:IsaacLabTutorialEnvCfg",
        "skrl_amp_cfg_entry_point": f"{agents.__name__}:skrl_amp_cfg.yaml",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:PPORunnerCfg",
    },
)

gym.register(
    id="Limo-NoNoise",
    entry_point=f"{__name__}.isaac_lab_tutorial_env:IsaacLabTutorialEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.isaac_lab_tutorial_env_cfg:LimoEnvCfg",
        "skrl_amp_cfg_entry_point": f"{agents.__name__}:skrl_amp_cfg.yaml",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_limo_ppo_cfg.yaml",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:PPORunnerCfg",
    },
)

gym.register(
    id="Limo-Noise005",
    entry_point=f"{__name__}.isaac_lab_tutorial_env:IsaacLabTutorialEnvWithNoise005",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.isaac_lab_tutorial_env_cfg:LimoEnvCfg",
        "skrl_amp_cfg_entry_point": f"{agents.__name__}:skrl_amp_cfg.yaml",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_limo_ppo_cfg.yaml",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:PPORunnerCfg",
    },
)

gym.register(
    id="Limo-Noise01",
    entry_point=f"{__name__}.isaac_lab_tutorial_env:IsaacLabTutorialEnvWithNoise01",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.isaac_lab_tutorial_env_cfg:LimoEnvCfg",
        "skrl_amp_cfg_entry_point": f"{agents.__name__}:skrl_amp_cfg.yaml",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_limo_ppo_cfg.yaml",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:PPORunnerCfg",
    },
)

gym.register(
    id="Limo-Noise02",
    entry_point=f"{__name__}.isaac_lab_tutorial_env:IsaacLabTutorialEnvWithNoise02",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.isaac_lab_tutorial_env_cfg:LimoEnvCfg",
        "skrl_amp_cfg_entry_point": f"{agents.__name__}:skrl_amp_cfg.yaml",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_limo_ppo_cfg.yaml",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:PPORunnerCfg",
    },
)

gym.register(
    id="Limo-Pendulum-NoNoise",
    entry_point=f"{__name__}.isaac_lab_tutorial_env:LimoPendulumNoNoiseEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.isaac_lab_tutorial_env_cfg:LimoPendulumEnvCfg",
        "skrl_amp_cfg_entry_point": f"{agents.__name__}:skrl_amp_cfg.yaml",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_limo_pendulum_cfg.yaml",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:PPORunnerCfg",
    },
)

gym.register(
    id="Limo-Pendulum-NoNoise-R2-64",
    entry_point=f"{__name__}.isaac_lab_tutorial_env:LimoPendulumNoNoiseEnv2",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.isaac_lab_tutorial_env_cfg:LimoPendulumEnvCfg",
        "skrl_amp_cfg_entry_point": f"{agents.__name__}:skrl_amp_cfg.yaml",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_limo_pendulum_cfg.yaml",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:PPORunnerCfg",
    },
)

gym.register(
    id="Limo-Pendulum-NoNoise-R2-32",
    entry_point=f"{__name__}.isaac_lab_tutorial_env:LimoPendulumNoNoiseEnv2",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.isaac_lab_tutorial_env_cfg:LimoPendulumEnvCfg",
        "skrl_amp_cfg_entry_point": f"{agents.__name__}:skrl_amp_cfg.yaml",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_limo_pendulum_cfg.yaml",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:PPORunnerCfg32",
    },
)

gym.register(
    id="Limo-Pendulum-NoNoise-R2-128",
    entry_point=f"{__name__}.isaac_lab_tutorial_env:LimoPendulumNoNoiseEnv2",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.isaac_lab_tutorial_env_cfg:LimoPendulumEnvCfg",
        "skrl_amp_cfg_entry_point": f"{agents.__name__}:skrl_amp_cfg.yaml",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_limo_pendulum_cfg.yaml",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:PPORunnerCfg128",
    },
)

gym.register(
    id="Limo-Pendulum-NoNoise-R2-32-16",
    entry_point=f"{__name__}.isaac_lab_tutorial_env:LimoPendulumNoNoiseEnv2",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.isaac_lab_tutorial_env_cfg:LimoPendulumEnvCfg",
        "skrl_amp_cfg_entry_point": f"{agents.__name__}:skrl_amp_cfg.yaml",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_limo_pendulum_cfg.yaml",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:PPORunnerCfg32_16",
    },
)