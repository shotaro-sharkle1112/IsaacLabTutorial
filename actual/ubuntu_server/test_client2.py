#!/usr/bin/env python3
"""
テストクライアント

推論サーバー(inference_server.py) に対して観測値を送信し、
返ってきた action を表示するだけの簡易クライアント。

使い方:
    python test_client.py
    python test_client.py --host 192.168.0.10 --port 5000
    python test_client.py --obs 0.1 -0.2 0.0 0.3
"""

import argparse
import json
import sys
from typing import List

import requests


def parse_args():
    parser = argparse.ArgumentParser(description="Inference Server Test Client")
    parser.add_argument(
        "--host", type=str, default="127.0.0.1",
        help="推論サーバーのホスト (デフォルト: 127.0.0.1)"
    )
    parser.add_argument(
        "--port", type=int, default=5000,
        help="推論サーバーのポート (デフォルト: 5000)"
    )
    parser.add_argument(
        "--obs", type=float, nargs="*",
        help="観測値(長さ4)をスペース区切りで指定 (例: --obs 0.1 -0.2 0.0 0.3)"
    )
    return parser.parse_args()


OBS_DIM = 4


def input_observation_from_stdin() -> List[float]:
    """
    標準入力から観測値を受け取る (対話モード用)
    """
    while True:
        line = input(f"観測値を {OBS_DIM} 個スペース区切りで入力してください: ")
        try:
            parts = line.strip().split()
            if len(parts) != OBS_DIM:
                print(f"要素数が {OBS_DIM} ではありません (len={len(parts)})。もう一度。")
                continue
            obs = [float(x) for x in parts]
            return obs
        except ValueError:
            print("float に変換できない値が含まれています。もう一度。")


def main():
    args = parse_args()

    url = f"http://{args.host}:{args.port}/infer"
    print(f"送信先URL: {url}")

    # 観測値を取得
    if args.obs is not None and len(args.obs) > 0:
        if len(args.obs) != OBS_DIM:
            print(f"[エラー] --obs には長さ {OBS_DIM} のベクトルを指定してください。")
            print(f"例: --obs 0.1 -0.2 0.0 0.3")
            sys.exit(1)
        obs = args.obs
    else:
        # 対話入力モード
        obs = input_observation_from_stdin()

    payload = {
        "observation": obs
    }

    print(f"送信する JSON: {json.dumps(payload)}")

    try:
        resp = requests.post(url, json=payload, timeout=5.0)
    except requests.exceptions.RequestException as e:
        print(f"[HTTPエラー] {e}")
        sys.exit(1)

    print(f"ステータスコード: {resp.status_code}")
    try:
        data = resp.json()
    except json.JSONDecodeError:
        print("サーバーからJSON以外の応答が返されました:")
        print(resp.text)
        sys.exit(1)

    print("サーバーの応答(JSON):")
    print(json.dumps(data, indent=2, ensure_ascii=False))

    if "action" in data:
        print(f"\n推論された action: {data['action']}")
    else:
        print("\n'action' フィールドが応答に含まれていません。")


if __name__ == "__main__":
    main()
