#!/usr/bin/env python3
"""
Playback of trajectories and storing them into a databaseself.
"""
from skills_manager.risk_aware_lfd.ralfd import InteractiveRALfD
from std_srvs.srv import Trigger
import rospy
import numpy as np 
from panda_ros.pose_transform_functions import array_quat_2_pose

try: # video_safety_layer package independency
    from video_embedding.utils import set_session
except ModuleNotFoundError:
    set_session = lambda: None

if __name__ == '__main__':
    name_skill = rospy.get_param('/execute_node/name_skill', "peg_door")
    localize_box = rospy.get_param('/execute_node/localize_box', True)
    session = rospy.get_param("/execute_node/session", "")
    set_session(session)

    print(f"Executing skill: {name_skill}, session: {session}")
    print("Localize box: ", localize_box)    
    lfd = InteractiveRALfD()

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
    lfd.loop()
    

