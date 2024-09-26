#!/usr/bin/env python3

import cv2
import numpy as np
from video_embedding.utils import load, set_session, visualize_labelled_video, visulize_video, load_video
import argparse


def main(args):
    set_session(args.session)
    # images = load_video(name=args.video)
    # print(len(images))
    # visulize_video(images)

    
    data = load(file=args.video)
    images= data['img']
    images_new=np.zeros((len(images),64,64))

    for i in range(len(images)):
        images_new[i]=cv2.resize(images[i], (64, 64))

    # images_new = images_new[:, np.newaxis, :, :]

    visualize_labelled_video(images_new, 
                             labels={
                                'risk_flag': data['risk_flag'],
                                'safe_flag': data['safe_flag'],
                            })


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="Play video",
        description="",
        epilog="",
    )
    parser.add_argument(
        "--video",
        default="peg_door_trial_2",
    )
    parser.add_argument(
        "--session",
        default="",
    )
    main(parser.parse_args())
