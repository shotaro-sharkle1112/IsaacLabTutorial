#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rospy
import math
from std_msgs.msg import Float32
from geometry_msgs.msg import Twist

class PendulumPIDNode:
    def __init__(self):
        rospy.init_node("pendulum_pid")

        # ====== パラメータ ======
        # 角度を読むトピック（pendulum_angle_nodeで出してる想定）
        self.angle_topic = rospy.get_param("~angle_topic", "/pendulum/angle_rad")
        # cmd_velの出力先
        self.cmd_vel_topic = rospy.get_param("~cmd_vel_topic", "/cmd_vel")

        # 目標角度（ラジアン）0ならまっすぐ
        self.target_angle = rospy.get_param("~target_angle", 0.0)

        # PIDゲイン
        self.kp = rospy.get_param("~kp", 2.0)
        self.ki = rospy.get_param("~ki", 0.0)
        self.kd = rospy.get_param("~kd", 0.2)

        # 出力制限
        self.max_vel = rospy.get_param("~max_vel", 0.5)  # [m/s]
        self.max_i   = rospy.get_param("~max_i", 1.0)    # 積分の上限（アンチワインドアップ）

        # 制御周期 [Hz]
        self.ctrl_hz = rospy.get_param("~ctrl_hz", 100.0)

        # 測定の微分にノイズが乗るならここをちょっと大きめにするとマイルドになる
        self.deriv_filter = rospy.get_param("~deriv_filter", 0.0)  # 0～1で一次フィルタ

        # ====== Pub/Sub ======
        self.sub = rospy.Subscriber(self.angle_topic, Float32, self.angle_cb, queue_size=1)
        self.pub = rospy.Publisher(self.cmd_vel_topic, Twist, queue_size=1)

        # ====== 内部状態 ======
        self.angle = None
        self.prev_angle = None
        self.integral = 0.0
        self.prev_time = rospy.get_time()
        self.prev_deriv = 0.0

        # ループ開始
        self.timer = rospy.Timer(rospy.Duration(1.0/self.ctrl_hz), self.control_cb)

        rospy.on_shutdown(self.on_shutdown)
        rospy.loginfo("pendulum_pid started. angle_topic=%s, cmd_vel_topic=%s",
                      self.angle_topic, self.cmd_vel_topic)

    def angle_cb(self, msg: Float32):
        # 最新角度を保存（ラジアン前提）
        self.angle = msg.data

    def control_cb(self, event):
        now = rospy.get_time()
        dt = now - self.prev_time
        if dt <= 0.0:
            return
        self.prev_time = now

        if self.angle is None:
            # まだ角度が来ていない
            return

        # ===== PID =====
        # 誤差（目標 - 現在）
        err = self.target_angle - self.angle

        # P
        p_term = self.kp * err

        # I（アンチワインドアップ）
        self.integral += err * dt
        # 積分のクリップ
        if self.integral > self.max_i:
            self.integral = self.max_i
        elif self.integral < -self.max_i:
            self.integral = -self.max_i
        i_term = self.ki * self.integral

        # D（角度の変化でとる。d(err)/dtでもいいけど、ここでは-measurement由来でも可）
        if self.prev_angle is None:
            d_term = 0.0
        else:
            # 角速度 [rad/s]（実際は振子の角速度）
            raw_deriv = (self.prev_angle - self.angle) / dt  # 測定値ベース
            # 簡単な一次フィルタ
            if self.deriv_filter > 0.0:
                raw_deriv = (1.0 - self.deriv_filter) * raw_deriv + self.deriv_filter * self.prev_deriv
            self.prev_deriv = raw_deriv
            d_term = self.kd * raw_deriv
        self.prev_angle = self.angle

        # PID出力. 符号は「前に倒れたら前へ」で合わせてる
        u = p_term + i_term + d_term

        # 速度としてクリップ
        if u > self.max_vel:
            u = self.max_vel
        elif u < -self.max_vel:
            u = -self.max_vel

        # ===== publish =====
        twist = Twist()
        twist.linear.x = u
        twist.angular.z = 0.0  # とりあえず今回はxだけ
        self.pub.publish(twist)

    def on_shutdown(self):
        # 停止を投げておく
        twist = Twist()
        self.pub.publish(twist)
        rospy.sleep(0.1)
        self.pub.publish(twist)

if __name__ == "__main__":
    try:
        node = PendulumPIDNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
