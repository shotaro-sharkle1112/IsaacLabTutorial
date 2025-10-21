#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math
import rospy
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Imu
from std_msgs.msg import Float32

try:
    from tf.transformations import euler_from_quaternion
    _HAS_TF = True
except Exception:
    _HAS_TF = False

class LimoPingPong:
    def __init__(self):
        rospy.init_node('limo_ping_pong')

        # ===== パラメータ =====
        self.speed   = rospy.get_param('~speed', 0.5)      # [m/s]
        self.period  = rospy.get_param('~period', 1.0)     # [s] 方向切替間隔
        self.imu_topic = rospy.get_param('~imu_topic', '/imu')
        self.yaw_topic = rospy.get_param('~yaw_topic', '/yaw')   # 出力先(ラジアン)
        self.log_yaw  = rospy.get_param('~log_yaw', True)        # 1秒毎にログ表示
        self.print_deg = rospy.get_param('~print_degrees', True) # ログ表示は度数法
        # 目標ヘディング（度指定）とデッドバンド（度）
        self.target_yaw_deg = rospy.get_param('~target_yaw_deg', 0.0)   # 例: 北=0°
        self.yaw_range_deg  = rospy.get_param('~yaw_range_deg', 5.0)    # 誤差5°以内なら回転0
        self.kp      = rospy.get_param('~kp', 0.4)                      # 比例ゲイン
        self.max_ang = rospy.get_param('~max_ang', 0.6)                  # 角速度上限[rad/s]  
        # 内部ではラジアンに
        self.target_yaw = math.radians(self.target_yaw_deg)
        self.yaw_range  = math.radians(self.yaw_range_deg)

        # ===== Publisher / Subscriber =====
        self.cmd_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=1)
        self.yaw_pub = rospy.Publisher(self.yaw_topic, Float32, queue_size=1)
        self.imu_sub = rospy.Subscriber(self.imu_topic, Imu, self._imu_cb, queue_size=1)

        # ===== 内部状態 =====
        self.dir = 1  # 1: 前進, -1: 後進
        self.cmd = Twist()
        self.yaw = None  # 最新Yaw[rad]

        # ===== タイマ =====
        rospy.Timer(rospy.Duration(self.period), self._toggle_cb)   # 方向反転
        rospy.Timer(rospy.Duration(0.05), self._publish_cb)         # 20Hz /cmd_vel
        if self.log_yaw:
            rospy.Timer(rospy.Duration(1.0), self._log_yaw_cb)      # 1Hz ログ

        rospy.on_shutdown(self._stop)

    # --- 1秒ごとに進行方向を反転 ---
    def _toggle_cb(self, event):
        self.dir *= -1

    # --- 20Hzで速度指令を出し続ける（誤差に基づく回頭） ---
    def _publish_cb(self, event):
       self.cmd.linear.x = self.dir * self.speed
 
       corr = 0.0
       if self.yaw is not None:
          # 誤差（target - current）を [-pi, pi] に正規化
          err = self._wrap_pi(self.target_yaw - self.yaw)
 
          # デッドバンド
          if abs(err) >= self.yaw_range:
                corr = self.kp * err
                # 上限クリップ（急激な回頭を防ぐ）
                corr = max(min(corr, self.max_ang), -self.max_ang)

       self.cmd.angular.z = corr
       self.cmd_pub.publish(self.cmd)

    # --- IMUコールバック: クォータニオン→Yaw(rad) ---
    def _imu_cb(self, msg: Imu):
        qx = msg.orientation.x
        qy = msg.orientation.y
        qz = msg.orientation.z
        qw = msg.orientation.w

        # 無効クォータニオン対策（オールゼロ等）
        norm = math.sqrt(qx*qx + qy*qy + qz*qz + qw*qw)
        if norm < 1e-6:
            return
        qx, qy, qz, qw = qx/norm, qy/norm, qz/norm, qw/norm

        if _HAS_TF:
            _, _, yaw = euler_from_quaternion([qx, qy, qz, qw])  # roll, pitch, yaw
        else:
            # 依存無しのYaw算出（Z軸回り, REP103 ENU準拠）
            # yaw = atan2(2(wz + xy), 1 - 2(y^2 + z^2))
            yaw = math.atan2(2.0*(qw*qz + qx*qy), 1.0 - 2.0*(qy*qy + qz*qz))

        self.yaw = self._wrap_pi(yaw)
        # トピック出力（ラジアン）
        self.yaw_pub.publish(Float32(self.yaw))

    # --- 1HzでYawをログ表示 ---
    def _log_yaw_cb(self, event):
        if self.yaw is None:
            return
        if self.print_deg:
            rospy.loginfo("Yaw: %.2f deg", math.degrees(self.yaw))
        else:
            rospy.loginfo("Yaw: %.3f rad", self.yaw)

    @staticmethod
    def _wrap_pi(a):
        # [-pi, pi] に正規化
        while a > math.pi:
            a -= 2.0*math.pi
        while a < -math.pi:
            a += 2.0*math.pi
        return a

    def _stop(self):
        stop = Twist()
        # 停止を数回送って確実に止める
        for _ in range(3):
            self.cmd_pub.publish(stop)
            rospy.sleep(0.05)

if __name__ == '__main__':
    try:
        LimoPingPong()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
