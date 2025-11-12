import yaml
import numpy as np
import gym
from rl_games.torch_runner import Runner

# ★ここを自分の実際のパスに変える
CONFIG_PATH = "infer_cfg.yaml"
CHECKPOINT_PATH = "last_limo-pendulum_ep_10000_rew_45.83476.pth"

# ★ここを学習環境に合わせて設定
OBS_DIM = 4   # 観測ベクトル次元
ACT_DIM = 1    # 行動次元（連続アクション想定）

# 1. 元のyaml読み込み
with open(CONFIG_PATH, "r") as f:
    cfg = yaml.safe_load(f)

import re

# ログファイルのパス
LOG_PATH = "infer_log.txt"

obs_list = []

pattern = re.compile(
    r"\[DEBUG\]: obs tensor\(\[([^\]]+)\]",  # [ ... ] の中身だけ取る
)

with open(LOG_PATH, "r") as d:
    for line in d:
        m = pattern.search(line)
        if m:
            # " 0.0734, -0.0573,  0.0016,  0.2029" → [0.0734, -0.0573, 0.0016, 0.2029]
            vals = [float(x.strip()) for x in m.group(1).split(",")]
            obs_list.append(vals)

obs_array = np.array(obs_list, dtype=np.float32)

action_list = []

pattern = re.compile(
    r"\[DEBUG\]: action tensor\(\[([^\]]+)\]",  # [ ... ] の中身だけ取る
)

with open(LOG_PATH, "r") as d:
    for line in d:
        m = pattern.search(line)
        if m:
            # " 0.0734, -0.0573,  0.0016,  0.2029" → [0.0734, -0.0573, 0.0016, 0.2029]
            vals = [float(x.strip()) for x in m.group(1).split(",")]
            action_list.append(vals)

action_array = np.array(action_list, dtype=np.float32)

# 2. envを作らせないための env_info を注入
conf = cfg["params"]["config"]

clip_obs = 2.0
clip_actions = 1.0

conf["env_info"] = {
    "observation_space": gym.spaces.Box(
        low=-np.inf,
        high=np.inf,
        shape=(OBS_DIM,),
        dtype=np.float32,
    ),
    "action_space": gym.spaces.Box(
        low=-clip_actions,
        high=clip_actions,
        shape=(ACT_DIM,),
        dtype=np.float32,
    ),
}


# プレイヤー側の設定を安全寄りに
player_conf = conf.setdefault("player", {})
player_conf["use_vecenv"] = False
player_conf["games_num"] = 1
player_conf["render"] = False

# 3. Runner & Player を作成
runner = Runner()
runner.load(cfg)

player = runner.create_player()
player.restore(CHECKPOINT_PATH)
player.reset()  # RNN使ってる場合の初期化
print("------------------------")
print(player.model)

state_dict = player.model.state_dict()
print("------------------------")
w = state_dict["sac_network.actor.trunk.2.weight"]
print("actor trunk2 weight shape:", w.shape)
print("actor trunk2 weight sample:\n", w[:3, :5])
print("------------------------")
print("has running_mean_std:", hasattr(player.model, "running_mean_std"))
if hasattr(player.model, "running_mean_std"):
    for n, p in player.model.running_mean_std.named_parameters():
        print("running_mean_std.", n, p.shape, p.mean().item(), p.std().item())
print("------------------------")
def policy(obs_np: np.ndarray) -> np.ndarray:
    import torch

    if obs_np.ndim == 1:
        obs_np = obs_np[None, :]
    
    obs_t = torch.from_numpy(obs_np).float()
    print(obs_t)
    # ② モデルと同じ device に乗せる
    device = next(player.model.parameters()).device
    obs_t = obs_t.to(device)

    # ③ play.py と同じ deterministic 設定を使う
    action = player.get_action(obs_t, is_deterministic=True)

    # ④ numpy に戻す
    if isinstance(action, torch.Tensor):
        action = action.detach().cpu().numpy()

    return action




# テスト
if __name__ == "__main__":
    for i, obs in enumerate(obs_array):
        print("true action:",action_array[i][-1],"| infer:",policy(obs))
