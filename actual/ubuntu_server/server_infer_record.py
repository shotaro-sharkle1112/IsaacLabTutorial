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
from flask import Flask, request, jsonify, send_file
from rl_games.torch_runner import Runner
import io


# 設定
CONFIG_PATH = "rl_games_sac_cfg2.yaml"
CHECKPOINT_PATH = "last_limo-pendulum_ep_10000_rew_45.83476.pth"
OBS_DIM = 4
ACT_DIM = 1
CLIP_OBS = 2.0
CLIP_ACTIONS = 1.0
HOST = "0.0.0.0"  # すべてのネットワークインターフェースでリッスン
PORT = 5000

# =========================
# リプレイバッファ関連の追加
# =========================
# 経験タプルを保存するリスト
# (obs, action, torch.unsqueeze(rewards, 1), next_obs_processed, torch.unsqueeze(dones, 1))
REPLAY_BUFFER = []

# 1つ前の観測と行動を保持
last_obs_np = None      # np.ndarray shape: (OBS_DIM,)
last_action_np = None   # np.ndarray shape: (ACT_DIM,)

rew_scale_pole_pos = -1.0
rew_scale_pole_vel = -0.001
rew_scale_cart_vel = -0.001
rew_scale_cart_pos = -0.001

def compute_reward_from_obs(obs_np: np.ndarray) -> float:
    """
    最新の観測から報酬を計算する関数。

    必要に応じて「タスク固有の報酬」に書き換えてください。
    ひとまず 0.0 を返すダミー実装にしてあります。
    """
    # 例: 観測のL2ノルムを負の報酬にする場合
    # return -float(np.linalg.norm(obs_np))
    #[振子の角度、振子の角速度、limoのx偏位、limoのx方向の線形速度]
    total_reward = np.exp(rew_scale_pole_pos*(obs_np[0]**2))+rew_scale_pole_vel*(obs_np[1]**2)+rew_scale_cart_pos*obs_np[2]+rew_scale_cart_vel*obs_np[3]
    return total_reward


def store_experience(prev_obs_np: np.ndarray,
                     prev_action_np: np.ndarray,
                     next_obs_np: np.ndarray,
                     reward: float,
                     done: bool = False):
    """
    リプレイバッファに1ステップ分の経験を保存する。

    保存形式:
        (obs, action, torch.unsqueeze(rewards, 1),
         next_obs_processed, torch.unsqueeze(dones, 1))
    """

    # 何が来ても受けられるように torch.tensor に統一
    obs_t = torch.tensor(prev_obs_np, dtype=torch.float32).unsqueeze(0)       # (1, OBS_DIM)
    action_t = torch.tensor(prev_action_np, dtype=torch.float32).unsqueeze(0) # (1, ACT_DIM)
    next_obs_t = torch.tensor(next_obs_np, dtype=torch.float32).unsqueeze(0)  # (1, OBS_DIM)

    rewards = torch.tensor([reward], dtype=torch.float32)   # (1,)
    dones = torch.tensor([done], dtype=torch.bool)          # (1,)

    experience = (
        obs_t,
        action_t,
        torch.unsqueeze(rewards, 1),  # (1, 1)
        next_obs_t,
        torch.unsqueeze(dones, 1),    # (1, 1)
    )

    REPLAY_BUFFER.append(experience)



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

    # 何が来ても受けられるようにする（list, ndarray, numpy.float32 など）
    obs_t = torch.tensor(obs_np, dtype=torch.float32, device=device)
    obs_t = torch.clamp(obs_t, -CLIP_OBS, CLIP_OBS)

    action = player.get_action(obs_t, is_deterministic=player.is_deterministic)

    if isinstance(action, torch.Tensor):
        action = action.detach().cpu().numpy()

    action = np.clip(action, -CLIP_ACTIONS, CLIP_ACTIONS)
    return action



@app.route("/infer", methods=["POST"])
def infer():
    """推論エンドポイント"""
    global last_obs_np, last_action_np  # 1つ前のステップ情報を更新するため

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

        # numpy配列に変換
        obs_np = np.array(obs, dtype=np.float32)

        # next_obs_processed として使うためにクリップ
        next_obs_processed_np = np.clip(obs_np, -CLIP_OBS, CLIP_OBS)

        # ===== リプレイバッファへの記録 =====
        # 「最新の観測データ」が到着したタイミングで、
        # 1つ前の (obs, action) と今回の next_obs_processed から経験を作る
        if last_obs_np is not None and last_action_np is not None:
            # 報酬を最新の観測から計算
            reward = compute_reward_from_obs(next_obs_processed_np)
            done = False  # 指定通り False 固定

            # (前obs, 前action, reward, next_obs_processed, done) を保存
            store_experience(
                prev_obs_np=last_obs_np,
                prev_action_np=last_action_np,
                next_obs_np=next_obs_processed_np,
                reward=reward,
                done=done,
            )

        # ===== 推論 =====
        action = policy(obs_np)

        # 次回のステップのために今回の obs / action を保存
        last_obs_np = next_obs_processed_np.copy()
        last_action_np = action.copy()

        # 結果を返す
        return jsonify({
            "action": action.tolist(),
            "status": "success",
            "buffer_size": len(REPLAY_BUFFER),  # おまけで現在のバッファサイズも返す
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/health", methods=["GET"])
def health():
    """ヘルスチェックエンドポイント"""
    return jsonify({
        "status": "healthy",
        "model_device": str(device),
        "buffer_size": len(REPLAY_BUFFER),
    })

@app.route("/download_buffer", methods=["GET"])
def download_buffer():
    """
    リプレイバッファを .pt ファイルとしてダウンロードするエンドポイント
    torch.load でそのまま読み込める形式
    """
    if not REPLAY_BUFFER:
        # 空のときはエラーにしておく（好みで変えてOK）
        return jsonify({"error": "replay buffer is empty"}), 400

    # メモリ上にバイナリファイルを作る
    buf = io.BytesIO()
    torch.save(REPLAY_BUFFER, buf)
    buf.seek(0)

    return send_file(
        buf,
        mimetype="application/octet-stream",
        as_attachment=True,
        download_name="replay_buffer.pt",  # 保存されるファイル名
    )

@app.route("/buffer", methods=["GET"])
def get_buffer():
    data = []
    for (obs_t, action_t, rew_t, next_obs_t, done_t) in REPLAY_BUFFER:
        data.append({
            "obs": obs_t.squeeze(0).tolist(),
            "action": action_t.squeeze(0).tolist(),
            "reward": float(rew_t.squeeze().item()),
            "next_obs": next_obs_t.squeeze(0).tolist(),
            "done": bool(done_t.squeeze().item()),
        })
    return jsonify({"size": len(REPLAY_BUFFER), "buffer": data})

if __name__ == "__main__":
    print(f"推論サーバーを起動します: http://{HOST}:{PORT}")
    print(f"推論エンドポイント: POST http://{HOST}:{PORT}/infer")
    print(f"ヘルスチェック: GET http://{HOST}:{PORT}/health")
    app.run(host=HOST, port=PORT, debug=False)
