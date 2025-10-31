#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rospy
import math
import numpy as np
from std_msgs.msg import Float32
from geometry_msgs.msg import Twist

import onnxruntime as ort   # pip install onnxruntime

class LimoOnnxController:
    def __init__(self):
        rospy.init_node("limo_onnx_controller")

        # ===== パラメータ =====
        self.model_path = rospy.get_param("~model_path", "/home/agilex/models/limo_ctrl.onnx")
        # ONNXの入力名（モデルによって違う。分からなければ print(session.get_inputs()) で確認）
        self.input_names = rospy.get_param("~input_names",
                                           ["pend_angle", "pend_angvel", "x_pos", "x_vel"])
        # ONNXの出力名
        self.output_name = rospy.get_param("~output_name", None)  # Noneなら最初の出力
        # 車輪半径[m]
        self.wheel_radius = rospy.get_param("~wheel_radius", 0.1)
        # 最大速度[m/s]（安全のため）
        self.max_lin_x = rospy.get_param("~max_lin_x", 0.5)
        # 推論周期[Hz]
        self.ctrl_hz = rospy.get_param("~ctrl_hz", 50.0)
        # cmd_velの出力先
        self.cmd_vel_topic = rospy.get_param("~cmd_vel_topic", "/cmd_vel")

        # ===== ONNX 読み込み =====
        # CPUでOK。GPUにしたいなら providers=["CUDAExecutionProvider","CPUExecutionProvider"]
        self.session = ort.InferenceSession(self.model_path, providers=["CPUExecutionProvider"])
        rospy.loginfo("Loaded ONNX model from %s", self.model_path)

        # 入力を全部1本にまとめるかどうかを判断
        # ここでは「入力が1つで shape=(1,4) みたいなモデル」と「入力が4本あるモデル」の両方に対応する
        self.onnx_inputs = self.session.get_inputs()
        self.multi_input_mode = (len(self.onnx_inputs) > 1)

        # ===== ROS Pub/Sub =====
        self.cmd_pub = rospy.Publisher(self.cmd_vel_topic, Twist, queue_size=1)

        self.angle = None
        self.angle_vel = None
        self.x_pos = None
        self.x_vel = None

        # サブスクライブ
        rospy.Subscriber("/pendulum/angle_rad", Float32, self._angle_cb, queue_size=1)
        rospy.Subscriber("/pendulum/angle_vel_rad", Float32, self._angle_vel_cb, queue_size=1)
        rospy.Subscriber("/limo/x", Float32, self._x_cb, queue_size=1)
        rospy.Subscriber("/limo/x_vel", Float32, self._xvel_cb, queue_size=1)

        # 制御タイマ開始
        rospy.Timer(rospy.Duration(1.0 / self.ctrl_hz), self._timer_cb)

        rospy.on_shutdown(self._on_shutdown)

    # ===== Callbacks =====
    def _angle_cb(self, msg: Float32):
        self.angle = float(msg.data)

    def _angle_vel_cb(self, msg: Float32):
        self.angle_vel = float(msg.data)

    def _x_cb(self, msg: Float32):
        self.x_pos = float(msg.data)

    def _xvel_cb(self, msg: Float32):
        self.x_vel = float(msg.data)

    # ===== main timer =====
    def _timer_cb(self, event):
        # まだ全部そろってないときは出さない
        if (self.angle is None or
            self.angle_vel is None or
            self.x_pos is None or
            self.x_vel is None):
            return

        # ----- ONNX に渡す入力を作る -----
        # モデルが「4入力それぞれ別名」の場合
        if self.multi_input_mode:
            # 入力名をもとに辞書を作る
            # ユーザーがinput_namesを指定してる場合を優先して使う
            feeds = {}
            vals = [self.angle, self.angle_vel, self.x_pos, self.x_vel]
            for name, v in zip(self.input_names, vals):
                # ONNXは基本float32
                feeds[name] = np.array([[v]], dtype=np.float32)
        else:
            # 入力が1つだけで shape=(1,4) の想定
            data = np.array([[self.angle, self.angle_vel, self.x_pos, self.x_vel]],
                            dtype=np.float32)
            feeds = {self.onnx_inputs[0].name: data}

        # ----- 推論 -----
        if self.output_name is None:
            outputs = self.session.run(None, feeds)
            y = outputs[0]
        else:
            outputs = self.session.run([self.output_name], feeds)
            y = outputs[0]

        # y のshapeを想定： [1,1] または [1] または scalar
        wheel_omega = float(np.array(y).reshape(-1)[0])  # [rad/s]

        # ----- /cmd_vel へ変換 -----
        # 直進のみ → linear.x = ω * r
        lin_x = wheel_omega * self.wheel_radius

        # 制限
        if lin_x > self.max_lin_x:
            lin_x = self.max_lin_x
        elif lin_x < -self.max_lin_x:
            lin_x = -self.max_lin_x

        twist = Twist()
        twist.linear.x = lin_x
        twist.angular.z = 0.0
        self.cmd_pub.publish(twist)

    def _on_shutdown(self):
        # 停止を投げて終わる
        twist = Twist()
        self.cmd_pub.publish(twist)
        rospy.sleep(0.05)
        self.cmd_pub.publish(twist)

if __name__ == "__main__":
    try:
        node = LimoOnnxController()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
