#!/usr/bin/env python3
# calc_variance.py
import argparse
import math
import sys

def read_values(path):
    """data.txt から数値を読み込む（空行と # 始まりは無視）"""
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            try:
                yield float(s)
            except ValueError:
                print(f"警告: 数値に変換できない行を無視しました: {s}", file=sys.stderr)

def linear_variance(path):
    """Welford法で平均・分散（母/標本）を逐次計算"""
    n = 0
    mean = 0.0
    M2 = 0.0
    for x in read_values(path):
        n += 1
        delta = x - mean
        mean += delta / n
        delta2 = x - mean
        M2 += delta * delta2
    if n == 0:
        return {"n": 0, "mean": float("nan"), "var_pop": float("nan"), "var_sample": float("nan")}
    var_pop = M2 / n
    var_sample = M2 / (n - 1) if n > 1 else float("nan")
    return {"n": n, "mean": mean, "var_pop": var_pop, "var_sample": var_sample}

def circular_variance(path, radians=False):
    """
    角度データの円形統計: circular variance = 1 - R
    R = sqrt((sum cos)^2 + (sum sin)^2) / n
    """
    n = 0
    C = 0.0
    S = 0.0
    for v in read_values(path):
        theta = v if radians else math.radians(v)
        C += math.cos(theta)
        S += math.sin(theta)
        n += 1
    if n == 0:
        return {"n": 0, "mean_circ_deg": float("nan"), "R": float("nan"), "var_circ": float("nan")}
    R = math.hypot(C, S) / n
    mean_angle = math.atan2(S, C)  # ラジアン
    mean_deg = math.degrees(mean_angle)
    # [-180, 180) に正規化
    if mean_deg >= 180:
        mean_deg -= 360
    var_circ = 1.0 - R
    return {"n": n, "mean_circ_deg": mean_deg, "R": R, "var_circ": var_circ}

def main():
    parser = argparse.ArgumentParser(description="data.txt の角度データの分散を計算します。")
    parser.add_argument("--path", nargs="?", default="degree_data.txt", help="入力ファイル（各行に数値）")
    parser.add_argument("--circular", action="store_true", help="円形（角度）データとして circular variance を計算")
    parser.add_argument("--radians", action="store_true", help="入力がラジアン値の場合（--circular と併用）")
    args = parser.parse_args()

    if args.circular:
        res = circular_variance(args.path, radians=args.radians)
        print(f"件数 n        : {res['n']}")
        print(f"円形平均[deg] : {res['mean_circ_deg']}")
        print(f"結果ant長 R   : {res['R']}")
        print(f"circular分散  : {res['var_circ']}")
    else:
        res = linear_variance(args.path)
        print(f"件数 n          : {res['n']}")
        print(f"平均            : {res['mean']}")
        print(f"母分散 (σ^2)    : {res['var_pop']}")
        print(f"標本分散 (s^2)  : {res['var_sample']}")

if __name__ == "__main__":
    main()
