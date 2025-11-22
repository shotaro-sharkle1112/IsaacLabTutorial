#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rospy
import requests
from nav_msgs.msg import Odometry
from std_msgs.msg import Float32

# デフォルトのサーバーURL（必要ならパラメータで上書き）
SERVER_URL_DEFAULT = "http://192.168.3.12:5000"


class LimoPendulumInferenceNode(object):
    def __init__(self):
        # パラメータからサーバーURL / タイムアウト / 送信周期を設定可能に
        self.server_url = rospy.get_param("~server_url", SERVER_URL_DEFAULT)
        self.timeout = rospy.get_param("~timeout", 0.1)     # [s]
        rate_hz = rospy.get_param("~rate", 60.0)             # [Hz]

        # 内部状態（最新値を保持）
        self.x0 = None        # 初期位置 x
        self.x = 0.0          # Limo の x 偏位
        self.x_dot = 0.0      # x 方向速度
        self.theta = 0.0      # 振子角
        self.theta_dot = 0.0  # 振子角速度

        # Subscriber
        self.sub_odom = rospy.Subscriber(
            "/imu", Odometry, self.odom_callback, queue_size=10
        )
        self.sub_angle = rospy.Subscriber(
            "/pendulum/angle_rad", Float32, self.angle_callback, queue_size=10
        )
        self.sub_angle_vel = rospy.Subscriber(
            "/pendulum/angle_vel_rad", Float32, self.angle_vel_callback, queue_size=10
        )

        # 推論結果（action）を流す Publisher
        self.pub_action = rospy.Publisher(
            "/limo_pendulum/action", Float32, queue_size=10
        )

        # 一定周期で推論＋publish
        self.timer = rospy.Timer(
            rospy.Duration(1.0 / rate_hz), self.timer_callback
        )

        rospy.loginfo("LimoPendulumInferenceNode started. server_url=%s", self.server_url)

    # Odometry コールバック：x と x_dot を更新
    def odom_callback(self, msg):
        # 初回だけ初期位置を保持して偏位を計算
        if self.x0 is None:
            self.x0 = msg.pose.pose.position.x

        # 偏位 = 現在位置 - 初期位置
        self.x = msg.pose.pose.position.x - self.x0

        # x 方向速度
        self.x_dot = msg.twist.twist.linear.x

    # 振子角（Float32）
    def angle_callback(self, msg):
        self.theta = msg.data

    # 振子角速度（Float32）
    def angle_vel_callback(self, msg):
        self.theta_dot = msg.data

    # Flask サーバーに observation を送って action をもらう
    def call_inference(self, observation):
        url = self.server_url + "/infer"
        payload = {"observation": observation}
        headers = {"Content-Type": "application/json"}

        try:
            response = requests.post(
                url, json=payload, headers=headers, timeout=self.timeout
            )
            response.raise_for_status()
            data = response.json()
        except Exception as e:
            # サーバーが落ちている等のときは警告だけ出して None を返す
            rospy.logwarn_throttle(1.0, "Inference request failed: %s", str(e))
            return None

        if "action" not in data:
            rospy.logwarn_throttle(1.0, "Response has no 'action' key: %s", data)
            return None

        action = data["action"]

        # action がリスト or タプルなら先頭要素を使用（一次元を想定）
        if isinstance(action, (list, tuple)):
            if len(action) == 0:
                rospy.logwarn_throttle(1.0, "Empty 'action' list in response: %s", data)
                return None
            action_value = float(action[0])
        else:
            # スカラーと仮定
            action_value = float(action)

        return action_value

    # 一定周期で呼ばれる：観測→推論→結果を publish
    def timer_callback(self, event):
        # まだ Odometry を一度も受信していない場合は何もしない
        if self.x0 is None:
            return

        # 観測ベクトル (x, x_dot, theta, theta_dot)
        observation = [self.x, self.x_dot, self.theta, self.theta_dot]

        action_value = self.call_inference(observation)
        if action_value is None:
            return

        msg = Float32()
        msg.data = action_value
        self.pub_action.publish(msg)


if __name__ == "__main__":
    rospy.init_node("limo_pendulum_inference_node")
    node = LimoPendulumInferenceNode()
    rospy.spin()
