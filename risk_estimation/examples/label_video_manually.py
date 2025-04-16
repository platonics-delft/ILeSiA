#!/usr/bin/env python3
import cv2
from video_embedding.models.video_embedding_dataset import load_dataloader
from video_embedding.utils import clip_samples, get_session, set_session, tensor_image_to_cv2, load
import argparse
import numpy as np

# skill_manager Python package needs to be installed correctly, then:
# from skills_manager.scripts.feedback import Feedback
from skills_manager.feedback import RiskAwareFeedback
from video_embedding.utils import get_trajectory_path

def save(self, file='last'):
    np.savez(f"{get_trajectory_path()}/trajectories/{get_session()}/{file}.npz",
                traj=self['traj'],
                ori=self['ori'],
                grip=self['grip'],
                img=self['img'], 
                img_feedback_flag=self['img_feedback_flag'],
                spiral_flag=self['spiral_flag'],
                risk_flag=self['risk_flag'],
                safe_flag=self['safe_flag'],
                novel_risk_flag=self['novel_risk_flag'],
                novel_safe_flag=self['novel_safe_flag'],
            )

def label_video(args):
    set_session(args['session'])
    data = dict(load(file=args['video']))
    
    # Create VideoEmbedder, assign name, load model and data
    dataloader = load_dataloader(args['video'], batch_size=64)
    video_embedder = VideoEmbedder(latent_dim=12)
    video_embedder.name = args['video']
    video_embedder.load(name=args['video'])

    raf = RiskAwareFeedback()

    risk_flag = np.zeros((len(video_embedder.tensor_images)))
    safe_flag = np.zeros((len(video_embedder.tensor_images)))
    novel_risk_flag = np.zeros((len(video_embedder.tensor_images)))
    novel_safe_flag = np.zeros((len(video_embedder.tensor_images)))
    spiral_flag = np.zeros((len(video_embedder.tensor_images)))
    for n, image in enumerate(video_embedder.tensor_images):
        image = video_embedder.tensor_images[n : n + 1]
        
        img = tensor_image_to_cv2(video_embedder.tensor_images[n])
        img = np.tile(img, (10, 10))
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
        
        if args['label_novel_dataset']:
            novel_risk_flag[n] = raf.novel_risk_flag
            novel_safe_flag[n] = raf.novel_safe_flag
        else:
            risk_flag[n] = raf.risk_flag
            safe_flag[n] = raf.safe_flag
        spiral_flag[n] = raf.spiral_flag

    
    print("Risk flag array:")
    print(risk_flag)
    print(safe_flag)
    print(novel_risk_flag)
    print(novel_safe_flag)
    print("----------------") 

    if args['label_novel_dataset']:
        data['novel_risk_flag'] = np.array([risk_flag])
        data['novel_safe_flag'] = np.array([safe_flag])
    else:
        data['risk_flag'] = np.array([risk_flag])
        data['safe_flag'] = np.array([safe_flag])
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
        
        if novel_risk_flag[n]:
            novelty_label = 'Nov R'
        elif novel_safe_flag[n]:
            novelty_label = 'Nov S'
        else:
            novelty_label = ''


        img = tensor_image_to_cv2(video_embedder.tensor_images[n])
        img = np.tile(img, (10, 10))
        cv2.putText(img, risk_label, (0, 12), cv2.FONT_HERSHEY_SIMPLEX,
            0.5, (255, 0, 0), 1, 2)
        cv2.putText(img, novelty_label, (12, 0), cv2.FONT_HERSHEY_SIMPLEX,
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
    parser.add_argument("--video", type=str)
    parser.add_argument("--session", default="quantitative_study")
    parser.add_argument("--label_novel_dataset", action="store_true")
    parser.set_defaults(label_novel_dataset=False)
    
    label_video(vars(parser.parse_args()))
