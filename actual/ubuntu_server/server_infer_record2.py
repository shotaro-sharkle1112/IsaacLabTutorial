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
import json
import threading

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
# 経験ログファイル関連
# =========================
# 1行1サンプルの JSON Lines 形式で保存する
EXPERIENCE_LOG_PATH = "experience_log.jsonl"
log_lock = threading.Lock()
try:
    with open(EXPERIENCE_LOG_PATH, "r", encoding="utf-8") as f:
        EXPERIENCE_COUNT = sum(1 for line in f if line.strip())
except FileNotFoundError:
    EXPERIENCE_COUNT = 0

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
    """
    # [振子の角度、振子の角速度、limoのx偏位、limoのx方向の線形速度]
    total_reward = (
        np.exp(rew_scale_pole_pos * (obs_np[0] ** 2))
        + rew_scale_pole_vel * (obs_np[1] ** 2)
        + rew_scale_cart_pos * obs_np[2]
        + rew_scale_cart_vel * obs_np[3]
    )
    return float(total_reward)


def store_experience(prev_obs_np: np.ndarray,
                     prev_action_np: np.ndarray,
                     next_obs_np: np.ndarray,
                     reward: float,
                     done: bool = False):
    """
    経験を JSON Lines 形式のテキストファイルに追記する。

    保存内容:
        {
          "obs": [...],
          "action": [...],
          "reward": float,
          "next_obs": [...],
          "done": bool
        }
    """
    global EXPERIENCE_COUNT

    exp = {
        "obs": prev_obs_np.tolist(),
        "action": prev_action_np.tolist(),
        "reward": float(reward),
        "next_obs": next_obs_np.tolist(),
        "done": bool(done),
    }

    line = json.dumps(exp, ensure_ascii=False)

    # 複数リクエストを想定して簡単なロックを入れておく
    with log_lock:
        with open(EXPERIENCE_LOG_PATH, "a", encoding="utf-8") as f:
            f.write(line + "\n")
        EXPERIENCE_COUNT += 1


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

        # ===== 経験のテキストファイルへの記録 =====
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
            "buffer_size": EXPERIENCE_COUNT,  # 現在のログ行数（経験数）
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/health", methods=["GET"])
def health():
    """ヘルスチェックエンドポイント"""
    return jsonify({
        "status": "healthy",
        "model_device": str(device),
        "buffer_size": EXPERIENCE_COUNT,
    })


@app.route("/download_buffer", methods=["GET"])
def download_buffer():
    """
    テキストログ(JSONL)から REPLAY_BUFFER 互換の .pt ファイルを生成して返すエンドポイント。
    torch.load でそのまま読み込める形式（経験タプルのリスト）になります。
    """
    experiences = []

    try:
        with open(EXPERIENCE_LOG_PATH, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                d = json.loads(line)

                obs_np = np.array(d["obs"], dtype=np.float32)
                action_np = np.array(d["action"], dtype=np.float32)
                reward = float(d["reward"])
                next_obs_np = np.array(d["next_obs"], dtype=np.float32)
                done = bool(d["done"])

                obs_t = torch.tensor(obs_np, dtype=torch.float32).unsqueeze(0)
                action_t = torch.tensor(action_np, dtype=torch.float32).unsqueeze(0)
                rewards_t = torch.tensor([[reward]], dtype=torch.float32)  # (1, 1)
                next_obs_t = torch.tensor(next_obs_np, dtype=torch.float32).unsqueeze(0)
                dones_t = torch.tensor([[done]], dtype=torch.bool)        # (1, 1)

                experiences.append(
                    (obs_t, action_t, rewards_t, next_obs_t, dones_t)
                )
    except FileNotFoundError:
        return jsonify({"error": "experience log file not found"}), 400

    if not experiences:
        return jsonify({"error": "experience log is empty"}), 400

    # メモリ上にバイナリファイルを作る
    buf = io.BytesIO()
    torch.save(experiences, buf)
    buf.seek(0)

    return send_file(
        buf,
        mimetype="application/octet-stream",
        as_attachment=True,
        download_name="replay_buffer.pt",
    )


@app.route("/buffer", methods=["GET"])
def get_buffer():
    """
    テキストログ(JSONL)の中身を確認するエンドポイント
    デフォルトでは最後の100件だけ返す（?limit=100 などで変更可）
    """
    from collections import deque

    limit = request.args.get("limit", default=100, type=int)
    buf = deque(maxlen=limit)
    total = 0

    try:
        with open(EXPERIENCE_LOG_PATH, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                d = json.loads(line)
                total += 1
                buf.append(d)
    except FileNotFoundError:
        return jsonify({"size": 0, "returned": 0, "buffer": []})

    data = list(buf)
    return jsonify({
        "size": total,          # ファイル全体の件数
        "returned": len(data),  # 今回返した件数
        "buffer": data,
    })


if __name__ == "__main__":
    print(f"推論サーバーを起動します: http://{HOST}:{PORT}")
    print(f"推論エンドポイント: POST http://{HOST}:{PORT}/infer")
    print(f"ヘルスチェック: GET http://{HOST}:{PORT}/health")
    app.run(host=HOST, port=PORT, debug=False)
