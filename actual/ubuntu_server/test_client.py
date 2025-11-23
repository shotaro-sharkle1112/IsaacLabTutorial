#!/usr/bin/env python3
"""
Flask 推論サーバーの簡単な動作確認クライアント

使い方:
    1. 別ターミナルで inference_server.py を起動しておく
         python inference_server.py

    2. このスクリプトを実行
         python test_client.py
"""

import time
import requests

SERVER_URL = "http://192.168.11.4:5000"  # 必要なら 0.0.0.0 やポートを変えてください


def check_health():
    """ヘルスチェックエンドポイントを叩く"""
    try:
        resp = requests.get(f"{SERVER_URL}/health")
        print("=== /health ===")
        print("status_code:", resp.status_code)
        print("response   :", resp.json())
    except Exception as e:
        print("ヘルスチェックに失敗しました:", e)

def check_download(filename="replay_buffer.pt"):
    """バッファーエンドポイントを叩いて .pt ファイルとして保存"""
    try:
        resp = requests.get(f"{SERVER_URL}/download_buffer")
        print("=== /download_buffer ===")
        print("status_code:", resp.status_code)

        if resp.status_code == 200:
            # 中身はバイナリなのでそのままファイルに書き出す
            with open(filename, "wb") as f:
                f.write(resp.content)
            print("saved buffer to:", filename)
        else:
            # エラー時はサーバー側が JSON を返しているはずなので json() してOK
            try:
                print("response   :", resp.json())
            except Exception:
                print("response(raw):", resp.text)

    except Exception as e:
        print("バッファーチェックに失敗しました:", e)

def check_buffer():
    try:
        resp = requests.get(f"{SERVER_URL}/buffer")
        print("=== /buffer ===")
        print("status_code:", resp.status_code)
        print("response   :", resp.json())
    except Exception as e:
        print("バッファーチェックに失敗しました:", e)


def call_infer(observation):
    """/infer に観測データを送って推論させる"""
    try:
        payload = {"observation": observation}
        resp = requests.post(f"{SERVER_URL}/infer", json=payload)

        print("=== /infer ===")
        print("sent obs    :", observation)
        print("status_code :", resp.status_code)
        print("response    :", resp.json())
    except Exception as e:
        print("推論リクエストに失敗しました:", e)


def main():
    # まずヘルスチェック
    check_health()

    # OBS_DIM = 4 を想定したテスト観測値
    test_observations = [
        [0.0, 0.0, 0.0, 0.0],
        [0.1, -0.1, 0.2, -0.2],
        [0.3, 0.4, -0.5, 0.6],
    ]

    # 複数回 /infer を叩いて、buffer_size が 2 回目以降で増えるかを見る
    for i, obs in enumerate(test_observations, start=1):
        print(f"\n--- step {i} ---")
        call_infer(obs)
        time.sleep(0.1)  # ログ見やすくするためにちょっとだけ待つ
    check_download()
    check_buffer()
    # 最後にもう一度ヘルスチェックして、buffer_size を確認
    print("\n最後に /health を再チェック")
    check_health()


if __name__ == "__main__":
    main()
