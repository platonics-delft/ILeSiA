#!/usr/bin/env python3
"""
Recording trajectories and storing them into a databaseself.
"""
from skills_manager.risk_aware_lfd.ralfd import RALfD
import rospy

try: # video_safety_layer package independency
    from video_embedding.utils import set_session
except ModuleNotFoundError:
    set_session = lambda: None
    
def main():
    try:
        session = rospy.get_param("/recording_node/session", "")
        name_skill = rospy.get_param("/recording_node/name_skill", )
        set_session(session)
        print(f"Recording skill: {name_skill}. Using session: {session}")

        lfd = RALfD()
        lfd.traj_rec()
        lfd.save(name_skill)
    except rospy.ROSInterruptException:
        pass

if __name__ == '__main__':
    main()
