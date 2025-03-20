#!/usr/bin/env python3
"""
Playback of trajectories and storing them into a databaseself.
"""
from video_embedding.utils import set_session
from skills_manager.risk_aware_lfd.ralfd import RALfD
from std_srvs.srv import Trigger
import sys
import os
import rospy    
import numpy as np 
from panda_ros.pose_transform_functions import array_quat_2_pose


if __name__ == '__main__':
    session = rospy.get_param('/execute_node/session')
    set_session(session)
    name_skill = rospy.get_param('/execute_node/name_skill')
    
    localize_box = rospy.get_param('/execute_node/localize_box')
    risk_policy = rospy.get_param('/execute_node/risk_policy')
    print("Executing skill: ", name_skill)
    print("Localize box: ", localize_box)
    print("Risk policy: ", risk_policy)
    lfd = RALfD(risk_policy)

    position = rospy.get_param("position")
    orientation = rospy.get_param("orientation") 

    pos_array = np.array([position['x'], position['y'], position['z']])
    quat = np.quaternion(orientation['w'], orientation['x'], orientation['y'], orientation['z'])
    goal = array_quat_2_pose(pos_array, quat)
    goal.header.seq = 1
    goal.header.stamp = rospy.Time.now()
    lfd.go_to_pose(goal)
    
    if localize_box:
        rospy.wait_for_service('active_localizer')
        active_localizer = rospy.ServiceProxy('active_localizer', Trigger)
        resp = active_localizer()
        lfd.compute_final_transform() 

    lfd.load(name_skill)
    lfd.execute()


    # If something is labelled, it is saved
    if sum(lfd.exec_record['risk_flag'].squeeze()) > 0 or sum(lfd.exec_record['safe_flag'].squeeze()) > 0:
        lfd.save(name_skill, risk_exec_trial=True)



