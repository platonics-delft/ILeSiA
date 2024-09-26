
import argparse
from copy import deepcopy
import numpy as np
import risk_estimation
from video_embedding.models.video_embedder import VideoEmbedder
from risk_estimation.plot_utils import plot_camera_images_along_robot_configurations
from video_embedding.utils import load
import spatialmath as sm

import roboticstoolbox as rtb
from spatialmath import SE3, UnitQuaternion
from franka_easy_ik import FrankaEasyIK

def main(args):
    data = load(args.video)
    # TODO: Add joint configurations data to demonstration 
    ik = FrankaEasyIK()
    qs = []

    for pos,ori in zip(data['traj'].T, data['ori'].T):
        qs.append(ik(p=pos, q=[ori[1], ori[2], ori[3], ori[0]]))

    plot_camera_images_along_robot_configurations(data['img'], qs, name=args.video)    


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="Play video",
        description="",
        epilog="",
    )
    parser.add_argument(
        "--video",
        default="peg_door",
    )
    main(parser.parse_args())
