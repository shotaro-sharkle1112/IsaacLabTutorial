import math

def _as_xyzw(q):
    """
    q: (x,y,z,w) のタプル/リスト か、{'x':..,'y':..,'z':..,'w':..} の辞書を受け付ける
    """
    if isinstance(q, dict):
        return (q['x'], q['y'], q['z'], q['w'])
    x, y, z, w = q  # 期待: (x,y,z,w)
    return (x, y, z, w)

def yaw_from_quat_xyzw(q):
    """
    ROS準拠の (x,y,z,w) からヨー角 [rad] を取り出す
    yaw = atan2(2(wz + xy), 1 - 2(y^2 + z^2))
    """
    x, y, z, w = _as_xyzw(q)
    return math.atan2(2.0*(w*z + x*y), 1.0 - 2.0*(y*y + z*z))

def angle_wrap_rad(a):
    """[-pi, pi) にラップ"""
    return (a + math.pi) % (2.0*math.pi) - math.pi

# --- 1) クォータニオン2つ＋走行距離 → 曲率 ---
def curvature_from_quaternions(q_start, q_end, s_m, b_m=None):
    """
    q_start, q_end: (x,y,z,w) または dict
    s_m: 走行距離 [m]
    b_m: トレッド幅 [m]（曲率計算には未使用。保持のみ）
    戻り値: (kappa [1/m], delta_yaw [rad])
    """
    if s_m <= 0:
        raise ValueError("s_m must be > 0")
    yaw0 = yaw_from_quat_xyzw(q_start)
    yaw1 = yaw_from_quat_xyzw(q_end)
    dtheta = angle_wrap_rad(yaw1 - yaw0)  # ヨー角の差をラップ
    kappa = dtheta / s_m
    return kappa, dtheta

# --- 2) 直線からのズレ y と走行距離 s → 曲率（小角近似） ---
def curvature_from_lateral_offset(y_m, s_m):
    """
    y_m: ゴール位置の横方向ズレ [m]（左+ / 右-）
    s_m: 指令した直進距離 [m]
    戻り値: kappa [1/m]
    近似: κ ≈ 2y / s^2 （|y| << s の小角前提）
    """
    if s_m <= 0:
        raise ValueError("s_m must be > 0")
    return 2.0 * y_m / (s_m * s_m)

# ---------- 使い方サンプル ----------
if __name__ == "__main__":
    # 例1: IMUのorientation（x,y,z,w）
    q0 = (0.0, 0.0, -0.0017453283659, 0.999998476913)  # 走行開始
    q1 = (0.0, 0.0, 0.0218148850346, 0.99976202708)  # 走行後
    s  = 2.0  # [m]
    b  = 0.175  # [m]（曲率には未使用）

    kappa1, dtheta = curvature_from_quaternions(q0, q1, s, b)
    print(f"[Quat] kappa = {kappa1:.8f} 1/m,  Δθ = {math.degrees(dtheta):.4f} deg")

    # 例2: 横ズレ y を使う（小角近似）
    y = 0.12  # [m]
    kappa2 = curvature_from_lateral_offset(y, s)
    print(f"[Offset] kappa ≈ {kappa2:.8f} 1/m  (small-angle)")
