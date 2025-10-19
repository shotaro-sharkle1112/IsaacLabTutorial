# ファイル名: pendulum_angle.py
import time, math
from mpu6050 import mpu6050

ADDR = 0x68
BUS  = 0      # あなたの環境は bus0
ALPHA = 0.98  # コンプリメンタリ係数（0.95〜0.99で調整）
SAMPLE_HZ = 200.0

def mean(vs): return sum(vs)/len(vs)

def rms(xs):
    m = mean(xs)
    return math.sqrt(sum((x-m)*(x-m) for x in xs)/len(xs))

def main():
    imu = mpu6050(ADDR, bus=BUS)
    # 推奨: 範囲とフィルタ（無ければコメントアウト）
    try:
        imu.set_accel_range(imu.ACCEL_RANGE_2G)
        imu.set_gyro_range(imu.GYRO_RANGE_250DEG)
        imu.set_filter_range(imu.FILTER_BW_42)
    except Exception:
        pass

    # --- 1) ジャイロ・バイアス校正（2秒：静止） ---
    print("Calibrating gyro offset... (keep still ~2s)")
    gx_buf, gy_buf = [], []
    t_end = time.time() + 2.0
    while time.time() < t_end:
        g = imu.get_gyro_data()     # deg/s
        gx_buf.append(g['x'])
        gy_buf.append(g['y'])
        time.sleep(1.0/400.0)
    gx_bias = mean(gx_buf)
    gy_bias = mean(gy_buf)
    print(f"Gyro bias: gx={gx_bias:.3f} dps, gy={gy_bias:.3f} dps")

    # --- 2) 倒れる方向の軸を自動判定（1秒：軽く傾ける） ---
    print("Detecting axis... (gently tilt in falling direction ~1s)")
    dx, dy = [], []
    t_end = time.time() + 1.0
    while time.time() < t_end:
        g = imu.get_gyro_data()
        dx.append(g['x'] - gx_bias)
        dy.append(g['y'] - gy_bias)
        time.sleep(1.0/400.0)
    axis = 'x' if rms(dx) >= rms(dy) else 'y'
    print(f"Selected axis: {axis.upper()}")

    # --- 3) 角度推定ループ ---
    # 初期角度は加速度から
    a0 = imu.get_accel_data(g=True)   # g単位
    if axis == 'x':
        # 回転軸X（ロール相当）: φ = atan2(Ay, Az)
        acc_angle_deg = math.degrees(math.atan2(a0['y'], a0['z']))
    else:
        # 回転軸Y（ピッチ相当）: θ = atan2(-Ax, sqrt(Ay^2 + Az^2))
        acc_angle_deg = math.degrees(math.atan2(-a0['x'], math.sqrt(a0['y']**2 + a0['z']**2)))
    angle_deg = acc_angle_deg

    print("Streaming angle [deg]  (Ctrl-C to quit)")
    dt_target = 1.0 / SAMPLE_HZ
    t_prev = time.time()

    while True:
        t_now = time.time()
        dt = t_now - t_prev
        t_prev = t_now

        # 1) ジャイロ角速度（バイアス補正済）で積分
        g = imu.get_gyro_data()  # deg/s
        rate_deg = (g['x'] - gx_bias) if axis == 'x' else (g['y'] - gy_bias)
        gyro_pred = angle_deg + rate_deg * dt

        # 2) 加速度から傾き（外乱が大きいので補助的に）
        a = imu.get_accel_data(g=True)  # g単位
        if axis == 'x':
            acc_angle_deg = math.degrees(math.atan2(a['y'], a['z']))
        else:
            acc_angle_deg = math.degrees(math.atan2(-a['x'], math.sqrt(a['y']**2 + a['z']**2)))

        # 3) コンプリメンタリフィルタ
        angle_deg = ALPHA * gyro_pred + (1.0 - ALPHA) * acc_angle_deg

        print(f"{angle_deg:+.2f}")
        # 目標サンプリング周期に合わせる（緩め）
        sleep_t = dt_target - (time.time() - t_now)
        if sleep_t > 0: time.sleep(sleep_t)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nBye")
