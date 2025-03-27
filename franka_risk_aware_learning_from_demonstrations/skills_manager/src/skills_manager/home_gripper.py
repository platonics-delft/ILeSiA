#!/usr/bin/env python3
import rospy
from panda_ros import Panda
import sys

if __name__ == "__main__":

    rospy.init_node("homing_node")
    panda=Panda()
    panda.home_gripper()
