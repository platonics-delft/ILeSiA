#!/usr/bin/env python3
import cv2
from video_embedding.models.video_embedding_dataset import load_dataloader
from video_embedding.utils import set_session, tensor_image_to_cv2, load
import argparse

def label_video(args):
    set_session(args['session'])
    data = dict(load(file=args['video']))
    
    dataloader = load_dataloader(args['video'], batch_size=64)
    tensor_images = dataloader.dataset.tensors[0][:, 0:1, :, :]

    for n, image in enumerate(tensor_images):
        
        if data['risk_flag'][0][n]:
            risk_label = 'R'
        elif data['safe_flag'][0][n]:
            risk_label = 'S'
        else:
            risk_label = ''
        
        if data['novel_risk_flag'][0][n]:
            novelty_label = 'Nov R'
        elif data['novel_safe_flag'][0][n]:
            novelty_label = 'Nov S'
        else:
            novelty_label = ''

        img = tensor_image_to_cv2(image)
        cv2.putText(img, risk_label, (0, 12), cv2.FONT_HERSHEY_SIMPLEX,
            0.5, (255, 0, 0), 1, 2)
        cv2.putText(img, novelty_label, (0, 62), cv2.FONT_HERSHEY_SIMPLEX,
            0.5, (255, 0, 0), 1, 2)

        cv2.namedWindow("Image", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Image", 640, 640)
        zoomed_image = cv2.resize(img, (640, 640), interpolation=cv2.INTER_NEAREST)
        cv2.imshow("Image", zoomed_image)
        if cv2.waitKey(25) & 0xFF == 27:  # Press 'Esc' to exit
            break
        # cv2.waitKey(0)  # Wait for a key press to close the window
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=str, default="peg_pick404")
    parser.add_argument("--session", default="quantitative_study")
    label_video(vars(parser.parse_args()))
