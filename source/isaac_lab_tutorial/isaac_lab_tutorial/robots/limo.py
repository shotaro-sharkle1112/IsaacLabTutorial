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
    prim_path="/World/envs/env_.*/robot",  # envクローン対応
    spawn=sim_utils.UsdFileCfg(
        usd_path=_LIMO_USD,
    ),
    actuators={"wheel_acts": ImplicitActuatorCfg(joint_names_expr=[".*"], damping=None, stiffness=None)},
)
