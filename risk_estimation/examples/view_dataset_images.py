import cv2
import risk_estimation
from risk_estimation.models.risk_estimation.frame_dropping import NoFrameDroppingPolicy, OnlyLabelledFramesDroppingPolicy
from risk_estimation.models.risk_estimation.risk_dataloader import RiskEstimationDataset
from risk_estimation.models.risk_estimation.risk_feature_extractor import LatentObservationsRiskLabels, VideoObservationsRiskAndSafeLabels
import video_embedding
from video_embedding.models.video_embedder import RiskyBehavioralVideoEmbedder
from video_embedding.utils import all_trial_names, clip_samples, number_of_saved_trials, set_session, tensor_image_to_cv2, visulize_video, load, visualize_labelled_video
import argparse
import numpy as np

# skill_manager Python package needs to be installed correctly, then:
from skills_manager.feedback import RiskAwareFeedback

def main(args):
    set_session(args.session)
    # for k in data.keys():
    #     print(f"{k}: {data[k].shape}")
    #     print(data[k])
    # print(len(images))
    # visulize_video(images)
    # data = clip_samples(data)
    # if input("Save cropped video? (y)") == 'y':
    #     save(data, file=args.video)
    
    video_embedder = RiskyBehavioralVideoEmbedder(name=args.video, latent_dim=args.latent_dim)
    video_names = all_trial_names(args.video, include_repr=True)
    video_embedder.load(video_names)
    video_embedder.load_model()

    dataset = RiskEstimationDataset.load_dataset(video_names, video_embedder, frame_dropping_policy=NoFrameDroppingPolicy, features=VideoObservationsRiskAndSafeLabels)
    print(dataset.X.shape)

    # Each video marked as separate
    for video_name in video_names:
        print(f"New Video: {video_name}")
        dataset = RiskEstimationDataset.load_dataset([video_name], video_embedder, frame_dropping_policy=OnlyLabelledFramesDroppingPolicy, features=VideoObservationsRiskAndSafeLabels)
        print(dataset.X.shape)

        labels = {
            'risk_flag': dataset.Y.cpu().numpy()[:,0],
            'safe_flag': dataset.Y.cpu().numpy()[:,1],
        }

        # dataset has embedded video images
        visualize_labelled_video(dataset.X.cpu().numpy(), labels = labels, press_for_next_frame=True, printer=True)

    return
    # All in once
    dataset = RiskEstimationDataset.load_dataset(video_names, video_embedder, frame_dropping_policy=OnlyLabelledFramesDroppingPolicy, features=VideoObservationsRiskAndSafeLabels)
    print(dataset.X.shape)

    labels = {
        'risk_flag': dataset.Y.cpu().numpy()[:,0],
        'safe_flag': dataset.Y.cpu().numpy()[:,1],
    }

    # dataset has embedded video images
    visualize_labelled_video(dataset.X.cpu().numpy(), labels = labels, press_for_next_frame=True, printer=True)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="Play video",
        description="",
        epilog="",
    )
    parser.add_argument("--video", default="peg_pick")
    parser.add_argument("--session", default="")
    parser.add_argument("--latent_dim", default=16)

    main(parser.parse_args())
