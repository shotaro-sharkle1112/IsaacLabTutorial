#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rospy
from nav_msgs.msg import Odometry
from std_msgs.msg import Float32

class OdomToLimoX:
    def __init__(self):
        rospy.init_node("odom_to_limo_x")

        self.odom_topic = rospy.get_param("~odom_topic", "/odom")
        self.pub_x = rospy.Publisher("/limo/x", Float32, queue_size=10)
        self.pub_x_vel = rospy.Publisher("/limo/x_vel", Float32, queue_size=10)

        self.x0 = None
        rospy.Subscriber(self.odom_topic, Odometry, self.odom_cb, queue_size=10)

    def odom_cb(self, msg: Odometry):
        x = msg.pose.pose.position.x               # odom基準のx位置[m]
        vx = msg.twist.twist.linear.x              # 前後速度[m/s]

        if self.x0 is None:
            self.x0 = x    # 起動時の位置をゼロ基準にする

        x_rel = x - self.x0

        self.pub_x.publish(Float32(x_rel))
        self.pub_x_vel.publish(Float32(vx))

if __name__ == "__main__":
    try:
        node = OdomToLimoX()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
