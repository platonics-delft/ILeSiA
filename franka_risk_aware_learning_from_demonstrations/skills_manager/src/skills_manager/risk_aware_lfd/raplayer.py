



import time
from risk_estimation.models.risk_estimator import sample_and_save_on_video
import numpy as np
from risk_estimation.models.safety_layer import SafetyLayer
from video_embedding.utils import visualize_labelled_video_frame
from skills_manager.player import Player
import rospy
from panda_ros.pose_transform_functions import position_2_array, array_quat_2_pose, list_2_quaternion


class RiskAwarePlayer(Player):
    """Demonstration player with risk-awareness capabilities
    """    
    def __init__(self):
        super(RiskAwarePlayer, self).__init__()

    def execute(self, retry_insertion_flag=0):

        self.sl = SafetyLayer(skill_name = self.filename, model_not_found_is_ok=True)  

        replay = False 
        start = self.player_init()
        self.traj_rec_init()

        
        while self.time_index <( self.recorded_traj.shape[1]) and not rospy.is_shutdown() and not self.end:
            try:
                
                if self.player_step(start, retry_insertion_flag) == 'stop':
                    break
                
                self.traj_rec_step()
                
                system_risk_pred = self.sl.get_estimated_risk(self.get_observations())
                visualize_labelled_video_frame(self.curr_image, risk_flag=system_risk_pred)
                
                action, alpha = self.risk_policy.do(self, system_risk_pred, self.risk_flag)

                if action == 'continue':
                    assert abs(alpha - (self.time_index / self.recorded_traj.shape[1])) < 1e-5
                    pass
                elif action == 'quit':
                    assert alpha == 1.0
                    self.time_index = int(alpha * self.recorded_traj.shape[1])
                elif action == 'repeat':
                    assert alpha == 0.0
                    self.recovery_mission(alpha)
                    self.time_index = int(alpha * self.recorded_traj.shape[1])
                    print("trying again")
                elif action == 'repeat_from_scratch':
                    replay = True
                    break
                elif action == 'recover':
                    print(f"recover alpha is {alpha}")

                    self.time_index = int(alpha * self.recorded_traj.shape[1])
                    self.go_to_time_index(self.time_index, linear=True)
                    
                else: raise Exception()
            except rospy.ROSInterruptException:
                break

        if replay:
            self.time_index = 0
            self.execute(retry_insertion_flag)

        self.save()

    def go_to_time_index(self, time_index: int, linear: bool = False):
        """
        Args:
            time_index (int): target time index
            linear (bool, optional): Linear motion to time index, non-blocking. Defaults to False.
        """        
        quat_goal = list_2_quaternion(self.recorded_ori[:, time_index])
        goal = array_quat_2_pose(self.recorded_traj[:, time_index] + self.camera_correction, quat_goal)
        goal.header.seq = 1
        goal.header.stamp = rospy.Time.now()
        goal.header.frame_id = 'panda_link0'
        
        self.correct()

        if (self.recorded_gripper[0][time_index]-self.recorded_gripper[0][max(0,time_index-1)]) < -self.grip_open_width/2:
            self.grasp_gripper(self.recorded_gripper[0][time_index])
            time.sleep(0.1)

        if (self.recorded_gripper[0][time_index]-self.recorded_gripper[0][max(0,time_index-1)]) > self.grip_open_width/2:
            self.move_gripper(self.recorded_gripper[0][time_index])
            time.sleep(0.1)

        if linear:
            self.go_to_pose(goal)
        else:
            self.goal_pub.publish(goal)

 
    def recovery_mission(self, alpha: float, linear: bool = True):
        """

        Args:
            alpha (float): taget frame as normalized time phase (0,1)
            linear (bool, optional): Defaults to True.
                If true: Going back via linear motion
                If false: Going back via same path - going backwards
        """        

        target_time_index = int(alpha * self.recorded_traj.shape[1])
            
        if linear:
            self.go_to_time_index(target_time_index, linear=True)
        else:
            while (self.time_index != target_time_index) and not rospy.is_shutdown() and not self.end:
                try:
                    system_risk_pred = self.sl.get_estimated_risk(self.get_observations())

                    next_time_index = self.time_index + np.clip(int(target_time_index) - int(self.time_index), -1, 1, dtype=int)
                    at_target = self.time_index == target_time_index
                    print(f"Now time index: {self.time_index}, {target_time_index}, next_time index: {next_time_index}, action: {at_target}")

                    self.go_to_time_index(next_time_index)

                    self.time_index = next_time_index

                    visualize_labelled_video_frame(self.curr_image, risk_flag=system_risk_pred)
                except rospy.ROSInterruptException:
                    break


    def finished_correctly(self):
        
        if self.time_index == self.recorded_traj.shape[1]:
            return True
        else:
            return False
        
    def save(self):
        sample_and_save_on_video(
            video_name=self.sl.video_embedder.name,
            video_embedder=self.sl.video_embedder,
            risk_estimator=self.sl.risk_estimator,
            risk_estimator2=self.sl.risk_estimator2,
            features=self.sl.feature_extractor,
            train_dataloader = None,
            folder="autogen"
        )

class InteractivePlayer(RiskAwarePlayer):
    def __init__(self):
        super(InteractivePlayer, self).__init__()
        
        self.target_time_index = 0
        self.at_target_previously = True
    
    def set_stiffness_once(self, at_target):
        if at_target == self.at_target_previously:
            return
        else:
            print("setting stiffness!")
            if at_target:
                self.set_stiffness(0, 0, 250, 0, 0, 0, 0)
            else:
                self.set_stiffness(3000, 3000, 3000, 40, 40, 40, 0)

        self.at_target_previously = at_target

    def loop(self, skill_name):
        
        self.sl = SafetyLayer(skill_name = skill_name)  
        self.load(file = skill_name)

        replay = False 
        start = self.player_init()
        
        # Turn on cv2 camera window
        while not rospy.is_shutdown() and not self.end:
            try:
                o = self.get_observations()
                x, y = self.sl.feature_extractor.extract(o, self.sl.video_embedder)
                # sample
                pred, risk = self.sl.sample(x)
                print(f"pred: {pred}, risk {risk}")
                
                next_time_index = self.time_index + np.clip(int(self.target_time_index) - int(self.time_index), -1, 1, dtype=int)
                at_target = self.time_index == self.target_time_index
                print(f"Now time index: {self.time_index}, {self.target_time_index}, next_time index: {next_time_index}, action: {at_target}")

                self.set_stiffness_once(at_target)

                if not at_target:
                    self.go_to_time_index(next_time_index)
                else:
                    time.sleep(0.1)
                
                self.time_index = next_time_index

                # visualize image
                visualize_labelled_video_frame(self.curr_image, risk_flag=pred)

                
            except rospy.ROSInterruptException:
                break