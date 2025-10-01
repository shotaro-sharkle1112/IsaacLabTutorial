# SPDX-License-Identifier: BSD-3-Clause
import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.actuators import ImplicitActuatorCfg

# ⚠️ あなたの環境の LIMO USD に合わせて変更してください
_LIMO_USD = f"{ISAAC_NUCLEUS_DIR}/Robots/AgilexRobotics/limo/limo.usd"

# ✅ USD にある4つの車輪ジョイント名（実ファイルに合わせて必要なら微修正）
_FL = "front_left_wheel"
_FR = "front_right_wheel"
_RL = "rear_left_wheel"
_RR = "rear_right_wheel"

LIMO_CFG = ArticulationCfg(
    prim_path="/World/envs/env_.*/Limo",  # envクローン対応
    spawn=sim_utils.UsdFileCfg(
        usd_path=_LIMO_USD,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            rigid_body_enabled=True,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=100.0,
            enable_gyroscopic_forces=True,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=4,
            solver_velocity_iteration_count=0,
            sleep_threshold=0.005,
            stabilization_threshold=0.001,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.0),
        # yaw は env 側で設定。車輪は0初期化。
        joint_pos={_FL: 0.0, _FR: 0.0, _RL: 0.0, _RR: 0.0},
    ),
    actuators={
        "wheel_acts" : ImplicitActuatorCfg(
           joint_names_expr = [_FL,_FR,_RL,_RR],
           stiffness=None,
           damping=None
        )
    },
)
