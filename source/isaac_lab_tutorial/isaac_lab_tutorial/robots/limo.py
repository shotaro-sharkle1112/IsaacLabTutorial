# SPDX-License-Identifier: BSD-3-Clause
import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.actuators import ImplicitActuatorCfg

# ⚠️ あなたの環境の LIMO USD に合わせて変更してください
_LIMO_USD = r"C:\Users\xr\Downloads\limo.usd"

# ✅ USD にある4つの車輪ジョイント名（実ファイルに合わせて必要なら微修正）
_FL = "front_left_wheel"
_FR = "front_right_wheel"
_RL = "rear_left_wheel"
_RR = "rear_right_wheel"

LIMO_CFG = ArticulationCfg(
    prim_path="/World/envs/env_.*/robot",
    spawn=sim_utils.UsdFileCfg(usd_path=_LIMO_USD),
    actuators={
        "wheel_acts": ImplicitActuatorCfg(
            joint_names_expr=[".*wheel.*"],
            stiffness=0.0, damping=1.0e4, effort_limit_sim=400.0,
        ),
        "pend_passive": ImplicitActuatorCfg(
            joint_names_expr=["pendulum"],  # ←実名に
            stiffness=0.0, damping=0.0,   # 完全受動
        ),
    },
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.1)
    ),
)

LIMO_FRONT_CFG = ArticulationCfg(
    prim_path="/World/envs/env_.*/robot",
    spawn=sim_utils.UsdFileCfg(
        usd_path=_LIMO_USD,
    ),
    # TODO:前輪のみ動かせているけど、後輪が引きずってて全く動かない
    actuators={
        # 前輪のみを制御対象に。ここに並べた順番がアクションの次元順になる想定（FL→FR）
        "front_wheels": ImplicitActuatorCfg(
            joint_names_expr=[_FL, _FR],
            damping=None, stiffness=None
        ),
    },
)