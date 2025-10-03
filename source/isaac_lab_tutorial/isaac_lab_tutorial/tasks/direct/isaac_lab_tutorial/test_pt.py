# predict_from_skrl_checkpoint.py
import os
import numpy as np
import torch

from skrl.agents.torch.ppo import PPO
from skrl.memories.torch import RandomMemory
from skrl.models.torch import Model, DeterministicMixin, GaussianMixin
from skrl.resources.preprocessors.torch import RunningStandardScaler

# ========= 必要に応じて変更 =========
CHECKPOINT_PATH = "best_agent.pt"  # best_agent.pt へのパス
OBS_DIM = 3     # 観測ベクトルの次元（学習時と同じにする）
ACT_DIM = 2     # 行動ベクトルの次元（学習時と同じにする）
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# ===================================

# --- skrl が参照する最小限の "Space ライク" オブジェクト（GymなしでOK） ---
class _BoxLike:
    def __init__(self, shape):
        self.shape = tuple(shape)
        self.dtype = np.float32

observation_space = _BoxLike((OBS_DIM,))
action_space = _BoxLike((ACT_DIM,))

# --- 学習時設定どおりのモデル ---
class Policy(GaussianMixin, Model):
    def __init__(self, observation_space, action_space, device):
        Model.__init__(self, observation_space, action_space, device)
        GaussianMixin.__init__(
            self,
            clip_actions=False,    # config: False
            clip_log_std=True,     # config: True
            min_log_std=-20.0,
            max_log_std=2.0,
            initial_log_std=0.0,
            reduction="sum"
        )
        n_obs = observation_space.shape[0]
        n_act = action_space.shape[0]
        self.net = torch.nn.Sequential(
            torch.nn.Linear(n_obs, 64),
            torch.nn.ELU(),
            torch.nn.Linear(64, 32),
            torch.nn.ELU(),
        )
        self.mean = torch.nn.Linear(32, n_act)

    def compute(self, inputs, role):
        x = inputs["states"]
        if not torch.is_tensor(x):
            x = torch.as_tensor(x, dtype=torch.float32, device=self.device)
        h = self.net(x)
        mean = self.mean(h)
        return mean, self.log_std, {}

class Value(DeterministicMixin, Model):
    def __init__(self, observation_space, action_space, device):
        Model.__init__(self, observation_space, action_space, device)
        DeterministicMixin.__init__(self, clip_actions=False)
        n_obs = observation_space.shape[0]
        self.net = torch.nn.Sequential(
            torch.nn.Linear(n_obs, 64),
            torch.nn.ELU(),
            torch.nn.Linear(64, 32),
            torch.nn.ELU(),
            torch.nn.Linear(32, 1),
        )

    def compute(self, inputs, role):
        x = inputs["states"]
        if not torch.is_tensor(x):
            x = torch.as_tensor(x, dtype=torch.float32, device=self.device)
        v = self.net(x)
        return v, {}

# --- 前処理（学習時: RunningStandardScaler） ---
state_preproc = RunningStandardScaler()
value_preproc = RunningStandardScaler()

# --- エージェントを構築（学習時の PPO 設定に整合） ---
models = {
    "policy": Policy(observation_space, action_space, DEVICE),
    "value":  Value(observation_space, action_space, DEVICE),
}
memory = RandomMemory(memory_size=32, num_envs=1, device=DEVICE)  # rollouts=32 に合わせる

agent_cfg = {
    "rollouts": 32,
    "learning_epochs": 8,
    "mini_batches": 8,
    "discount_factor": 0.99,
    "lambda": 0.95,
    "learning_rate": 5.0e-04,
    "learning_rate_scheduler": "KLAdaptiveLR",
    "learning_rate_scheduler_kwargs": {"kl_threshold": 0.008},
    "state_preprocessor": state_preproc,
    "value_preprocessor": value_preproc,
    "random_timesteps": 0,
    "learning_starts": 0,
    "grad_norm_clip": 1.0,
    "ratio_clip": 0.2,
    "value_clip": 0.2,
    "clip_predicted_values": True,
    "entropy_loss_scale": 0.0,
    "value_loss_scale": 2.0,
    "kl_threshold": 0.0,
    "rewards_shaper_scale": 0.1,
    "time_limit_bootstrap": False,
}

agent = PPO(
    models=models,
    memory=memory,
    cfg=agent_cfg,
    observation_space=observation_space,
    action_space=action_space,
    device=DEVICE
)

# --- チェックポイント読み込み ---
if not os.path.isfile(CHECKPOINT_PATH):
    raise FileNotFoundError(f"checkpoint not found: {CHECKPOINT_PATH}")
agent.load(CHECKPOINT_PATH)
agent.set_running_mode("eval")
for m in models.values():
    m.eval()

# --- API: 観測(単発 or バッチ) → 行動を返す ---
@torch.no_grad()
def predict_actions(observations, deterministic=False):
    """
    observations: shape (obs_dim,) or (N, obs_dim), dtype float32/64
    deterministic=False: True のとき平均（mean）を返す。False のとき確率的サンプル。
    return: np.ndarray shape (N, act_dim)
    """
    obs = np.asarray(observations, dtype=np.float32)
    if obs.ndim == 1:
        obs = obs[None, :]

    if deterministic:
        # 決定論的：policy の平均値をそのまま出す（前処理は agent に合わせて適用）
        # state_preprocessor を通すために agent のユーティリティを使う
        # 内部仕様に依存しない簡潔版: agent.act を呼んでから平均だけ再計算する方法もあるが
        # ここでは models["policy"].compute を直接呼ぶ
        # RunningStandardScaler は agent 内部で使われるので、ここでは生データのまま渡す
        states_t = torch.from_numpy(obs).to(DEVICE)
        mean, log_std, _ = models["policy"].compute({"states": states_t}, role="policy")
        return mean.cpu().numpy()
    else:
        # 確率的：agent.act（前処理を内部適用）でサンプルを得る
        actions, _, _ = agent.act(states=obs, role="policy")
        # actions は torch.Tensor or np.ndarray。np にして返す
        if isinstance(actions, torch.Tensor):
            actions = actions.cpu().numpy()
        return actions

# ===== 使い方例 =====
if __name__ == "__main__":
    # 単発
    obs = np.zeros((OBS_DIM,), dtype=np.float32)
    a_det = predict_actions(obs, deterministic=True)   # 平均（決定論的）
    a_sto = predict_actions(obs, deterministic=False)  # サンプル（確率的）
    print("deterministic:", a_det)
    print("stochastic   :", a_sto)

    # バッチ
    obs_batch = np.stack([np.zeros(OBS_DIM, np.float32),
                          np.ones(OBS_DIM,  np.float32)*0.5], axis=0)
    acts = predict_actions(obs_batch, deterministic=False)
    print("batch actions:", acts)
