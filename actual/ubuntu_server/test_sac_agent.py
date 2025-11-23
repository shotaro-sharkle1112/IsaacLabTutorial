import yaml
import numpy as np
import gym
from rl_games.torch_runner import Runner
import torch


# ★ここを自分の実際のパスに変える
CONFIG_PATH = "rl_games_sac_cfg2.yaml"
CHECKPOINT_PATH = "limo-pendulum.pth"

# ★ここを学習環境に合わせて設定
OBS_DIM = 4   # 観測ベクトル次元
ACT_DIM = 1    # 行動次元（連続アクション想定）

# 1. 元のyaml読み込み
with open(CONFIG_PATH, "r") as f:
    cfg = yaml.safe_load(f)

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

runner = Runner()
runner.load(cfg)

agent = runner.create_agent()
agent.restore(CHECKPOINT_PATH)

agent.init_tensors()
agent.algo_observer.after_init(agent)


for i in range(5000):
    agent.replay_buffer.add(torch.tensor([[ 3.5487e-02,  0.0000e+00, -5.2387e-10,  0.0000e+00]], device='cuda:0') ,torch.tensor([[-0.3472]], device='cuda:0'),torch.tensor([[0.9801]], device='cuda:0') ,torch.tensor([[ 0.0183, -1.6309,  0.0336,  0.7934]], device='cuda:0') ,torch.tensor([[False]], device='cuda:0'))
print(agent.replay_buffer.sample(1))
print("------------------------")
print(agent.replay_buffer.capacity)
print("------------------------")
print(agent.model)

state_dict = agent.model.state_dict()
print("------------------------")
w = state_dict["sac_network.actor.trunk.2.weight"]
print("actor trunk2 weight shape:", w.shape)
print("actor trunk2 weight sample:\n", w[:3, :5])
print("------------------------")

agent.update(0)

state_dict = agent.model.state_dict()
print("------------------------")
w = state_dict["sac_network.actor.trunk.2.weight"]
print("actor trunk2 weight shape:", w.shape)
print("actor trunk2 weight sample:\n", w[:3, :5])
print("------------------------")





def policy(obs_np: np.ndarray) -> np.ndarray:


    if obs_np.ndim == 1:
        obs_np = obs_np[None, :]
    
    obs_t = torch.from_numpy(obs_np).float()
    # ② モデルと同じ device に乗せる
    device = next(agent.model.parameters()).device
    obs_t = obs_t.to(device)
    print(obs_t[0])
    # ③ play.py と同じ deterministic 設定を使う
    action = agent.act(obs_t, 1, sample=False)

    # ④ numpy に戻す
    if isinstance(action, torch.Tensor):
        action = action.detach().cpu().numpy()

    return action

obs = np.array([[ 0.0516, -1.1301, 0.0163, 0.6297],
                [ 0.0516, -1.1301, 0.0163, 0.6297]])
