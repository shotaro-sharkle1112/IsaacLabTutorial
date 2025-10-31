#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import time, math
import rospy
from std_msgs.msg import Float32
from mpu6050 import mpu6050

def mean(vs): return sum(vs)/len(vs)
def rms(xs):
    m = mean(xs)
    return math.sqrt(sum((x-m)*(x-m) for x in xs)/len(xs))

class PendulumAngleNode:
    def __init__(self):
        rospy.init_node("pendulum_angle")

        # ===== パラメータ =====
        self.addr = rospy.get_param("~addr", 0x68)
        self.bus = rospy.get_param("~bus", 0)
        self.alpha = float(rospy.get_param("~alpha", 0.98))              # complementaryの係数
        self.sample_hz = float(rospy.get_param("~sample_hz", 200.0))
        self.calib_sec = float(rospy.get_param("~calib_sec", 2.0))
        self.axis_detect_sec = float(rospy.get_param("~axis_detect_sec", 1.0))
        self.axis_param = str(rospy.get_param("~axis", "auto")).lower()

        # 出力トピック
        self.topic_deg = rospy.get_param("~topic_deg", "/pendulum/angle_deg")
        self.topic_rad = rospy.get_param("~topic_rad", "/pendulum/angle_rad")
        self.topic_vel_deg = rospy.get_param("~topic_vel_deg", "/pendulum/angle_vel_deg")
        self.topic_vel_rad = rospy.get_param("~topic_vel_rad", "/pendulum/angle_vel_rad")

        # 角速度の1次フィルタ量（0ならなし）
        self.vel_filter = float(rospy.get_param("~vel_filter", 0.0))
        # 追加: 角速度のデッドバンド（deg/s）。これより小さいときは0にする
        self.vel_deadband_deg = float(rospy.get_param("~vel_deadband_deg", 0.3))
        # 追加: 角度の微小変化を無視したいとき（deg）。0ならしない
        self.angle_deadband_deg = float(rospy.get_param("~angle_deadband_deg", 0.0))

        # ===== Publisher =====
        self.pub_deg = rospy.Publisher(self.topic_deg, Float32, queue_size=10)
        self.pub_rad = rospy.Publisher(self.topic_rad, Float32, queue_size=10)
        self.pub_vel_deg = rospy.Publisher(self.topic_vel_deg, Float32, queue_size=10)
        self.pub_vel_rad = rospy.Publisher(self.topic_vel_rad, Float32, queue_size=10)

        # ===== IMU初期化 =====
        self.imu = mpu6050(self.addr, bus=self.bus)
        try:
            self.imu.set_accel_range(self.imu.ACCEL_RANGE_2G)
            self.imu.set_gyro_range(self.imu.GYRO_RANGE_250DEG)
            self.imu.set_filter_range(self.imu.FILTER_BW_42)
        except Exception:
            pass

        # ===== 1) ジャイロバイアス校正 =====
        rospy.loginfo("Calibrating gyro offset... keep still for ~%.1fs", self.calib_sec)
        gx_buf, gy_buf = [], []
        t_end = time.time() + self.calib_sec
        while (not rospy.is_shutdown()) and time.time() < t_end:
            g = self.imu.get_gyro_data()  # deg/s
            gx_buf.append(g['x'])
            gy_buf.append(g['y'])
            time.sleep(1.0/400.0)
        self.gx_bias = mean(gx_buf) if gx_buf else 0.0
        self.gy_bias = mean(gy_buf) if gy_buf else 0.0
        rospy.loginfo("Gyro bias: gx=%.3f dps, gy=%.3f dps", self.gx_bias, self.gy_bias)

        # ===== 2) 軸判定 =====
        if self.axis_param in ("x", "y"):
            self.axis = self.axis_param
        else:
            rospy.loginfo("Detecting axis... gently tilt in falling direction for ~%.1fs", self.axis_detect_sec)
            dx, dy = [], []
            t_end = time.time() + self.axis_detect_sec
            while (not rospy.is_shutdown()) and time.time() < t_end:
                g = self.imu.get_gyro_data()
                dx.append(g['x'] - self.gx_bias)
                dy.append(g['y'] - self.gy_bias)
                time.sleep(1.0/400.0)
            self.axis = 'x' if (dx and dy and rms(dx) >= rms(dy)) else 'y'
        rospy.loginfo("Selected axis: %s", self.axis.upper())

        # ===== 3) 初期角度 =====
        a0 = self.imu.get_accel_data(g=True)
        if self.axis == 'x':
            acc_angle_deg = math.degrees(math.atan2(a0['y'], a0['z']))
        else:
            acc_angle_deg = math.degrees(math.atan2(-a0['x'], math.sqrt(a0['y']**2 + a0['z']**2)))
        self.angle_deg = acc_angle_deg
        self.angle_vel_deg = 0.0

        # ループ用
        self.dt_target = 1.0 / self.sample_hz
        self.t_prev = time.time()

    def spin(self):
        rate = rospy.Rate(self.sample_hz)
        rospy.loginfo("Streaming angle & angular velocity at %.1f Hz", self.sample_hz)

        while not rospy.is_shutdown():
            t_now = time.time()
            dt = t_now - self.t_prev
            if dt <= 0:
                dt = self.dt_target
            self.t_prev = t_now

            # 1) gyro読取り（バイアス除去）
            g = self.imu.get_gyro_data()  # deg/s
            if self.axis == 'x':
                rate_deg_raw = g['x'] - self.gx_bias
            else:
                rate_deg_raw = g['y'] - self.gy_bias

            # 2) gyro積分で予測
            gyro_pred = self.angle_deg + rate_deg_raw * dt

            # 3) accから角度
            a = self.imu.get_accel_data(g=True)
            if self.axis == 'x':
                acc_angle_deg = math.degrees(math.atan2(a['y'], a['z']))
            else:
                acc_angle_deg = math.degrees(math.atan2(-a['x'], math.sqrt(a['y']**2 + a['z']**2)))

            # 4) complementaryで角度更新（ここで一旦new_angleに）
            new_angle_deg = self.alpha * gyro_pred + (1.0 - self.alpha) * acc_angle_deg

            # 5) 角度のデッドバンド（オプション）
            if self.angle_deadband_deg > 0.0 and abs(new_angle_deg - self.angle_deg) < self.angle_deadband_deg:
                # 変化がすごく小さいときは前回値を保持
                pass
            else:
                self.angle_deg = new_angle_deg

            # 6) 角速度のフィルタ＆デッドバンド
            if self.vel_filter > 0.0:
                vel_deg = (1.0 - self.vel_filter) * rate_deg_raw + self.vel_filter * self.angle_vel_deg
            else:
                vel_deg = rate_deg_raw

            # デッドバンド適用
            if abs(vel_deg) < self.vel_deadband_deg:
                vel_deg = 0.0

            self.angle_vel_deg = vel_deg

            # 7) Publish
            self.pub_deg.publish(Float32(self.angle_deg))
            self.pub_rad.publish(Float32(math.radians(self.angle_deg)))
            self.pub_vel_deg.publish(Float32(self.angle_vel_deg))
            self.pub_vel_rad.publish(Float32(math.radians(self.angle_vel_deg)))

            try:
                rate.sleep()
            except rospy.ROSInterruptException:
                break

def main():
    node = PendulumAngleNode()
    node.spin()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nBye")
