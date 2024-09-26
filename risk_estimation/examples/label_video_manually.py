#!/usr/bin/env python3
import cv2
from video_embedding.models.video_embedder import VideoEmbedder
from video_embedding.utils import clip_samples, get_session, set_session, tensor_image_to_cv2, visulize_video, load
import argparse
import numpy as np

# skill_manager Python package needs to be installed correctly, then:
# from skills_manager.scripts.feedback import Feedback
from skills_manager.feedback import RiskAwareFeedback
import rospkg

def save(self, file='last'):
    ros_pack = rospkg.RosPack()
    _package_path = ros_pack.get_path('trajectory_data')
    np.savez(f"{_package_path}/trajectories/{get_session()}/{file}.npz",
                traj=self['traj'],
                ori=self['ori'],
                grip=self['grip'],
                img=self['img'], 
                img_feedback_flag=self['img_feedback_flag'],
                spiral_flag=self['spiral_flag'],
                risk_flag=self['risk_flag'],
                safe_flag=self['safe_flag'],
                novelty_flag=self['novelty_flag'],
                recovery_phase=self['recovery_phase'],)

def label_video(args):
    set_session(args['session'])
    data = dict(load(file=args['video']))
    if input("Save cropped video (400 frames)? (y)") == 'y':
        data = clip_samples(data)
        save(data, file=args['video'])

    # Create VideoEmbedder, assign name, load model and data
    video_embedder = VideoEmbedder(latent_dim=8)
    video_embedder.name = args['video']
    video_embedder.load(name=args['video'])

    raf = RiskAwareFeedback()

    risk_flag = np.zeros((len(video_embedder.tensor_images)))
    safe_flag = np.zeros((len(video_embedder.tensor_images)))
    novelty_flag = np.zeros((len(video_embedder.tensor_images)))
    recovery_phase = -1.0 * np.ones((len(video_embedder.tensor_images)))
    spiral_flag = np.zeros((len(video_embedder.tensor_images)))
    for n, image in enumerate(video_embedder.tensor_images):
        image = video_embedder.tensor_images[n : n + 1]
        
        img = tensor_image_to_cv2(video_embedder.tensor_images[n])
        img = cv2.resize(img, (640, 640))
        cv2.putText(
            img,
            '',
            (0, 12),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 0, 0),
            1,
            2,
        )

        cv2.namedWindow("Image", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Image", 1000, 500)
        cv2.imshow("Image", img)
        if cv2.waitKey(25) & 0xFF == 27:  # Press 'Esc' to exit
            break
        # cv2.waitKey(0)  # Wait for a key press to close the window
        
        risk_flag[n] = raf.risk_flag
        safe_flag[n] = raf.safe_flag
        novelty_flag[n] = raf.novelty_flag
        recovery_phase[n] = raf.recovery_phase
        spiral_flag[n] = raf.spiral_flag

    
    print("Risk flag array:")
    print(risk_flag)
    print(safe_flag)
    print(recovery_phase)
    print("----------------") 

    if args['only_label_recovery_phase']:
        data['recovery_phase'] = np.array([recovery_phase])
    else:
        data['risk_flag'] = np.array([risk_flag])
        data['safe_flag'] = np.array([safe_flag])
        data['novelty_flag'] = np.array([novelty_flag])
        data['recovery_phase'] = np.array([recovery_phase])
        data['spiral_flag'] = np.array([spiral_flag])


    print("Manual labelling ended, see the results")
    for n, image in enumerate(video_embedder.tensor_images):
        image = video_embedder.tensor_images[n : n + 1]
        
        if risk_flag[n]:
            risk_label = 'R'
        elif safe_flag[n]:
            risk_label = 'S'
        else:
            risk_label = ''
        
        if novelty_flag[n]:
            novelty_label = 'N'
        else:
            novelty_label = ''

        if recovery_phase[n] != -1.0:
            recovery_phase_label = str(round(recovery_phase[n], 1))
        else:
            recovery_phase_label = ""


        img = tensor_image_to_cv2(video_embedder.tensor_images[n])
        img = cv2.resize(img, (640, 640))
        cv2.putText(img, risk_label, (0, 12), cv2.FONT_HERSHEY_SIMPLEX,
            0.5, (255, 0, 0), 1, 2)
        cv2.putText(img, recovery_phase_label, (12, 0), cv2.FONT_HERSHEY_SIMPLEX,
            0.5, (255, 0, 0), 1, 2)


        cv2.namedWindow("Image", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Image", 1000, 500)
        cv2.imshow("Image", img)
        if cv2.waitKey(25) & 0xFF == 27:  # Press 'Esc' to exit
            break
        # cv2.waitKey(0)  # Wait for a key press to close the window
        
    if input("Save? (y)") == 'y':
        save(data, file=args['video'])

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="Video labeller",
        description="",
        epilog="",
    )
    parser.add_argument(
        "--video",
        default="peg_door_trial_6",
    )
    parser.add_argument(
        "--session",
        default="",
    )
    parser.add_argument("--only_label_recovery_phase", action="store_true")
    parser.add_argument("--label_all", dest="only_label_recovery_phase", action="store_false")
    parser.set_defaults(only_label_recovery_phase=True)
    
    label_video(vars(parser.parse_args()))
