
from risk_estimation.models.risk_estimation.frame_dropping import *


OnlyLabelledFramesDroppingPolicyRiskPegPick
OnlyLabelledFramesDroppingPolicyRiskPegDoor
OnlyLabelledFramesDroppingPolicyRiskPegPlace
OnlyLabelledFramesDroppingPolicyRiskSliderMove
OnlyLabelledFramesDroppingPolicyRiskMoveAround

import cv2, pathlib
from video_embedding.utils import load, get_session
import rospkg

mapping = {
    "peg_pick404": "PegPick", 
    "peg_door404": "PegDoor", 
    "slider_move404": "SliderMove", 
    "slider_move404_2": "SliderMove", # has better alignment between train and test trajectory data
    "peg_place404": "PegPlace", 
    "probe_pick404": "ProbePick",
    "move_around404": "MoveAround",
}

def get_labeled_section_frames(name_skill: str):
    ros_pack = rospkg.RosPack()
    _package_path = ros_pack.get_path('trajectory_data')
    
    frame_drop = eval(f"OnlyLabelledFramesDroppingPolicyRisk{mapping[name_skill]}")

    list_of_tensor_images = []
    l1 = frame_drop.maxtestcut - frame_drop.mintestcut
    l2 = frame_drop.maxtestcut2 - frame_drop.mintestcut2
    v = 0
    while pathlib.Path(f"{_package_path}/trajectories/{get_session()}/{name_skill}_test_{v}").is_file():
        name = f"{name_skill}_test_{v}"
        v+=1
        
        data = load(file=name)
        images= data['img']
        risk_flag = data['risk_flag']
        safe_flag = data['safe_flag']
        images_new=np.zeros((l1+l2,64,64))
        
        for n,i in enumerate(list(range(frame_drop.mintestcut,frame_drop.maxtestcut)) + list(range(frame_drop.mintestcut2,frame_drop.maxtestcut2))):
            images_new[n]=cv2.resize(images[i], (64, 64), interpolation=cv2.INTER_AREA)

        images = images_new[:, np.newaxis, :, :]
        tensor_images = torch.tensor(images, dtype=torch.float32).cuda()

        list_of_tensor_images.append(tensor_images)
    return list_of_tensor_images