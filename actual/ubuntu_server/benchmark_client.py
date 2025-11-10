#!/usr/bin/env python3
"""
推論サーバーのベンチマークテスト
1000個のダミーデータを送信して処理時間を計測

使用方法:
    python benchmark_client.py
"""
import requests
import json
import time
import numpy as np
from statistics import mean, stdev

# サーバーのURL
SERVER_URL = "http://192.168.3.12:5000"
NUM_REQUESTS = 1000
OBS_DIM = 4


def generate_dummy_observation():
    """ランダムなダミー観測データを生成"""
    return np.random.uniform(-2.0, 2.0, OBS_DIM).tolist()


def send_inference_request(observation):
    """推論リクエストを送信"""
    response = requests.post(
        f"{SERVER_URL}/infer",
        json={"observation": observation},
        headers={"Content-Type": "application/json"}
    )
    return response


def benchmark():
    """ベンチマークテスト"""
    print(f"=== 推論サーバー ベンチマークテスト ===")
    print(f"送信データ数: {NUM_REQUESTS}")
    print(f"サーバーURL: {SERVER_URL}\n")

    # ヘルスチェック
    try:
        health_response = requests.get(f"{SERVER_URL}/health", timeout=5)
        print(f"サーバーステータス: {health_response.json()}\n")
    except Exception as e:
        print(f"エラー: サーバーに接続できません - {e}")
        return

    # ダミーデータを事前生成
    print("ダミーデータを生成中...")
    dummy_observations = [generate_dummy_observation() for _ in range(NUM_REQUESTS)]
    print("生成完了\n")

    # ベンチマーク実行
    print(f"{NUM_REQUESTS}件の推論リクエストを送信中...")

    success_count = 0
    failure_count = 0
    response_times = []

    start_time = time.time()

    for i, obs in enumerate(dummy_observations):
        try:
            request_start = time.time()
            response = send_inference_request(obs)
            request_end = time.time()

            if response.status_code == 200:
                success_count += 1
                response_times.append(request_end - request_start)
            else:
                failure_count += 1

        except Exception as e:
            failure_count += 1
            print(f"リクエスト {i+1} でエラー: {e}")

        # 進捗表示（100件ごと）
        if (i + 1) % 100 == 0:
            print(f"  {i + 1}/{NUM_REQUESTS} 完了...")

    end_time = time.time()
    total_time = end_time - start_time

    # 結果表示
    print("\n" + "="*50)
    print("ベンチマーク結果")
    print("="*50)
    print(f"総リクエスト数: {NUM_REQUESTS}")
    print(f"成功: {success_count}")
    print(f"失敗: {failure_count}")
    print(f"\n総実行時間: {total_time:.2f} 秒")
    print(f"スループット: {success_count / total_time:.2f} req/sec")

    if response_times:
        print(f"\nレスポンスタイム統計:")
        print(f"  平均: {mean(response_times)*1000:.2f} ms")
        print(f"  最小: {min(response_times)*1000:.2f} ms")
        print(f"  最大: {max(response_times)*1000:.2f} ms")
        if len(response_times) > 1:
            print(f"  標準偏差: {stdev(response_times)*1000:.2f} ms")


if __name__ == "__main__":
    benchmark()
