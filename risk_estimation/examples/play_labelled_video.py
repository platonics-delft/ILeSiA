#!/usr/bin/env python3

from risk_estimation.models.risk_estimation.frame_dropping import NoFrameDroppingPolicy
from risk_estimation.models.risk_estimation.risk_dataloader import RiskEstimationDataset
from risk_estimation.models.safety_layer import SafetyLayer
import torch
from video_embedding.utils import set_session, load, visualize_labelled_video
import argparse

def main(args):
    set_session(args.session)
    data = dict(load(file=args.video))
    visualize_labelled_video(data['img'], data, printer=True)
    
    sl = SafetyLayer(skill_name=args.video)

    dataset = RiskEstimationDataset.load_dataset(video_names=[args.video], video_embedder=sl.video_embedder,frame_dropping_policy=NoFrameDroppingPolicy, features=sl.feature_extractor)

    X = dataset.X
    assert len(X) == len(data['img'])

    pred, risk = sl.sample(torch.tensor(X.cpu().numpy(), dtype=torch.long).cuda())        
    visualize_labelled_video(data['img'], {'risk_flag': pred}, printer=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="Play video",
        description="",
        epilog="",
    )
    parser.add_argument(
        "--video",
        default="look_at_door",
    )
    parser.add_argument(
        "--session",
        default="",
    )
    
    main(parser.parse_args())
