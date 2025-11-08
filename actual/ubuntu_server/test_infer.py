import yaml
import numpy as np
from rl_games.torch_runner import Runner

# 1. yamlを読む（runner.pyと同じ）
with open("path/to/config.yaml", "r") as f:
    cfg = yaml.safe_load(f)

runner = Runner()
runner.load(cfg)  # cfg は top-level に params を持つ想定
                   # (runner.py もこれと同じ流れ) :contentReference[oaicite:0]{index=0}

# 2. Player を生成
player = runner.create_player()

# 3. checkpoint を読み込み
player.restore("runs/xxx/nn/xxx.pth")  # players.PpoPlayer* の restore を呼ぶ :contentReference[oaicite:1]{index=1}
player.reset()  # RNNあり構成のときの状態初期化（無くても死にはしない）

# 4. 観測から行動を1ステップ出す
obs = np.array([...], dtype=np.float32)  # 学習時と同じ形・スケール
action = player.get_action(obs, is_deterministic=True)

print(action)