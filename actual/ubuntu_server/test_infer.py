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


player_conf = conf.setdefault("player", {})
player_conf["use_vecenv"] = False
player_conf["games_num"] = 1
player_conf["render"] = False
# play.py と同じにしたいなら deterministic=True/False をここで明示
player_conf.setdefault("deterministic", True)

runner = Runner()
runner.load(cfg)
player = runner.create_player()
player.restore(CHECKPOINT_PATH)
player.reset()  # RNN使ってないなら実害ほぼなし

def policy(obs_np: np.ndarray):
    # ★obs_npはそのまま渡す。normalizeもclipもplayer側がやる。
    #   shapeは (OBS_DIM,) または (1, OBS_DIM)
    return player.get_action(obs_np, is_deterministic=player.is_deterministic)




# テスト
if __name__ == "__main__":
    for i, obs in enumerate(obs_array):
        print("true action:",action_array[i][-1],"| infer:",policy(obs))
