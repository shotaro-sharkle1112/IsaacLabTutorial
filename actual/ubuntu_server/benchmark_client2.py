#!/usr/bin/env python3
"""
benchmark_client.py

推論サーバー(inference_server.py) に対して観測値を高速に送り、
送信速度・レイテンシをベンチマークするクライアント。

Example:
    python benchmark_client.py
    python benchmark_client.py -n 1000 -c 4 --host 192.168.0.10 --port 5000
"""

import argparse
import json
import time
from typing import List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests

OBS_DIM = 4


def parse_args():
    parser = argparse.ArgumentParser(description="Inference Server Benchmark Client")
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
        help=f"観測値(長さ {OBS_DIM})をスペース区切りで指定 (例: --obs 0.1 -0.2 0.0 0.3)。"
             f"未指定なら [0.0, 0.0, 0.0, 0.0] を使用。"
    )
    parser.add_argument(
        "-n", "--num-requests", type=int, default=100,
        help="送信するリクエスト総数 (デフォルト: 100)"
    )
    parser.add_argument(
        "-c", "--concurrency", type=int, default=1,
        help="同時接続数 (並列スレッド数, デフォルト: 1)"
    )
    parser.add_argument(
        "--warmup", type=int, default=5,
        help="ベンチマーク前のウォームアップリクエスト数 (デフォルト: 5)"
    )
    parser.add_argument(
        "--timeout", type=float, default=5.0,
        help="各リクエストのタイムアウト秒数 (デフォルト: 5.0)"
    )
    return parser.parse_args()


def prepare_observation(args_obs: Optional[List[float]]) -> List[float]:
    if args_obs is None or len(args_obs) == 0:
        return [0.0] * OBS_DIM
    if len(args_obs) != OBS_DIM:
        raise ValueError(f"--obs には長さ {OBS_DIM} のベクトルを指定してください。")
    return args_obs


def send_request(
    url: str,
    payload: dict,
    timeout: float,
) -> Tuple[bool, float]:
    """
    1回のHTTPリクエストを送信し、成功フラグとレイテンシ(sec)を返す。
    """
    t0 = time.perf_counter()
    try:
        resp = requests.post(url, json=payload, timeout=timeout)
        t1 = time.perf_counter()
        ok = resp.status_code == 200
        return ok, t1 - t0
    except requests.exceptions.RequestException:
        t1 = time.perf_counter()
        return False, t1 - t0


def worker(
    url: str,
    payload: dict,
    timeout: float,
    count: int,
) -> Tuple[int, int, List[float]]:
    """
    1つのスレッドで count 回リクエストを送る。
    戻り値: (成功数, 失敗数, レイテンシリスト[秒])
    """
    success = 0
    failure = 0
    latencies: List[float] = []

    for _ in range(count):
        ok, latency = send_request(url, payload, timeout)
        latencies.append(latency)
        if ok:
            success += 1
        else:
            failure += 1

    return success, failure, latencies


def print_stats(total: int, success: int, failure: int, latencies: List[float], elapsed: float):
    print("\n===== Benchmark Result =====")
    print(f"Total requests : {total}")
    print(f"Success        : {success}")
    print(f"Failure        : {failure}")
    print(f"Total time     : {elapsed:.4f} sec")

    if total > 0 and elapsed > 0:
        print(f"Requests/sec   : {total / elapsed:.2f} req/s")

    if latencies:
        latencies_sorted = sorted(latencies)
        avg = sum(latencies_sorted) / len(latencies_sorted)
        p50 = latencies_sorted[int(0.5 * (len(latencies_sorted) - 1))]
        p90 = latencies_sorted[int(0.9 * (len(latencies_sorted) - 1))]
        p99 = latencies_sorted[int(0.99 * (len(latencies_sorted) - 1))]
        print(f"\nLatency [sec]")
        print(f"  min : {latencies_sorted[0]:.6f}")
        print(f"  avg : {avg:.6f}")
        print(f"  p50 : {p50:.6f}")
        print(f"  p90 : {p90:.6f}")
        print(f"  p99 : {p99:.6f}")
        print(f"  max : {latencies_sorted[-1]:.6f}")


def main():
    args = parse_args()

    obs = prepare_observation(args.obs)
    url = f"http://{args.host}:{args.port}/infer"
    payload = {"observation": obs}

    print(f"送信先URL: {url}")
    print(f"観測値   : {obs}")
    print(f"総リクエスト数: {args.num_requests}, 並列数: {args.concurrency}, ウォームアップ: {args.warmup}")

    # ウォームアップ
    if args.warmup > 0:
        print(f"\n--- Warmup ({args.warmup} requests) ---")
        for i in range(args.warmup):
            ok, latency = send_request(url, payload, args.timeout)
            status = "OK" if ok else "NG"
            print(f"  [{i+1}/{args.warmup}] {status}, {latency:.4f} sec")

    # ベンチマーク本番
    n = args.num_requests
    c = max(1, args.concurrency)

    # スレッドごとに割り当てるリクエスト数を計算
    base = n // c
    extra = n % c
    counts = [base + (1 if i < extra else 0) for i in range(c)]
    counts = [cnt for cnt in counts if cnt > 0]  # 0 のスレッドは削除

    print("\n--- Benchmark start ---")
    print(f"スレッドごとのリクエスト数: {counts}")

    total_success = 0
    total_failure = 0
    all_latencies: List[float] = []

    start = time.perf_counter()

    if len(counts) == 1:
        # シングルスレッド
        s, f, lat = worker(url, payload, args.timeout, counts[0])
        total_success += s
        total_failure += f
        all_latencies.extend(lat)
    else:
        # マルチスレッド
        with ThreadPoolExecutor(max_workers=len(counts)) as ex:
            futures = []
            for cnt in counts:
                futures.append(
                    ex.submit(worker, url, payload, args.timeout, cnt)
                )

            for fut in as_completed(futures):
                s, f, lat = fut.result()
                total_success += s
                total_failure += f
                all_latencies.extend(lat)

    end = time.perf_counter()
    elapsed = end - start

    print_stats(n, total_success, total_failure, all_latencies, elapsed)


if __name__ == "__main__":
    main()
