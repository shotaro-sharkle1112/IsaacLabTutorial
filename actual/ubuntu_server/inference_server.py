#!/usr/bin/env python3
"""
推論サーバー
LANを通じて観測データを受け取り、推論結果を返すシンプルなHTTPサーバー

使用方法:
    python inference_server.py

エンドポイント:
    POST /infer
    Body: {"observation": [0.0734, -0.0573, 0.0016, 0.2029]}
    Response: {"action": [0.123]}
"""
import yaml
import numpy as np
import gym
import torch
from flask import Flask, request, jsonify
from rl_games.torch_runner import Runner

# 設定
CONFIG_PATH = "infer_cfg.yaml"
CHECKPOINT_PATH = "last_limo-pendulum_ep_10000_rew_45.83476.pth"
OBS_DIM = 4
ACT_DIM = 1
CLIP_OBS = 2.0
CLIP_ACTIONS = 1.0
HOST = "0.0.0.0"  # すべてのネットワークインターフェースでリッスン
PORT = 5000

# Flaskアプリの初期化
app = Flask(__name__)

# モデルのロード（サーバー起動時に1度だけ実行）
print("モデルをロード中...")
with open(CONFIG_PATH, "r") as f:
    cfg = yaml.safe_load(f)

conf = cfg["params"]["config"]
conf["env_info"] = {
    "observation_space": gym.spaces.Box(
        low=-CLIP_OBS,
        high=CLIP_OBS,
        shape=(OBS_DIM,),
        dtype=np.float32,
    ),
    "action_space": gym.spaces.Box(
        low=-CLIP_ACTIONS,
        high=CLIP_ACTIONS,
        shape=(ACT_DIM,),
        dtype=np.float32,
    ),
}

player_conf = conf.setdefault("player", {})
player_conf["use_vecenv"] = False
player_conf["games_num"] = 1
player_conf["render"] = False

runner = Runner()
runner.load(cfg)
player = runner.create_player()
player.restore(CHECKPOINT_PATH)
player.reset()

device = next(player.model.parameters()).device
print(f"モデルをロード完了 (device: {device})")


def policy(obs_np: np.ndarray) -> np.ndarray:
    """観測データから行動を推論"""
    obs_t = torch.from_numpy(obs_np).float()
    obs_t = torch.clamp(obs_t, -CLIP_OBS, CLIP_OBS)
    obs_t = obs_t.to(device)

    action = player.get_action(obs_t, is_deterministic=player.is_deterministic)

    if isinstance(action, torch.Tensor):
        action = action.detach().cpu().numpy()

    action = np.clip(action, -CLIP_ACTIONS, CLIP_ACTIONS)
    return action


@app.route("/infer", methods=["POST"])
def infer():
    """推論エンドポイント"""
    try:
        # リクエストからJSONデータを取得
        data = request.get_json()

        if "observation" not in data:
            return jsonify({"error": "observation フィールドが必要です"}), 400

        obs = data["observation"]

        # 観測データの検証
        if not isinstance(obs, list) or len(obs) != OBS_DIM:
            return jsonify({
                "error": f"observation は長さ {OBS_DIM} のリストである必要があります"
            }), 400

        # numpy配列に変換して推論
        obs_np = np.array(obs, dtype=np.float32)
        action = policy(obs_np)

        # 結果を返す
        return jsonify({
            "action": action.tolist(),
            "status": "success"
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/health", methods=["GET"])
def health():
    """ヘルスチェックエンドポイント"""
    return jsonify({"status": "healthy", "model_device": str(device)})


if __name__ == "__main__":
    print(f"推論サーバーを起動します: http://{HOST}:{PORT}")
    print(f"推論エンドポイント: POST http://{HOST}:{PORT}/infer")
    print(f"ヘルスチェック: GET http://{HOST}:{PORT}/health")
    app.run(host=HOST, port=PORT, debug=False)
