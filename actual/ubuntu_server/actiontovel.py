#!/usr/bin/env python
# -*- coding: utf-8 -*-

import rospy
from std_msgs.msg import Float32
from geometry_msgs.msg import Twist


class LimoPendulumActionToCmdVel(object):
    def __init__(self):
        # ノード初期化
        rospy.init_node('limo_pendulum_action_to_cmd_vel')

        # 定数倍のスケール（パラメータから変更可能、デフォルト 0.5）
        self.scale = rospy.get_param('~scale', 0.5)

        # Publisher: /cmd_vel (Twist)
        self.cmd_vel_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=1)

        # Subscriber: /limo_pendulum/action (Float32)
        self.sub = rospy.Subscriber(
            '/limo_pendulum/action',
            Float32,
            self.action_callback,
            queue_size=1
        )

        rospy.loginfo('limo_pendulum_action_to_cmd_vel started. scale = %f', self.scale)

    def action_callback(self, msg):
        # 受け取った値を定数倍
        scaled = msg.data * self.scale

        # Twist メッセージ作成
        cmd = Twist()
        cmd.linear.x = scaled
        # 他の成分は 0 のまま

        # パブリッシュ
        self.cmd_vel_pub.publish(cmd)

        rospy.logdebug('action = %.3f -> cmd_vel.linear.x = %.3f', msg.data, scaled)


if __name__ == '__main__':
    try:
        node = LimoPendulumActionToCmdVel()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
