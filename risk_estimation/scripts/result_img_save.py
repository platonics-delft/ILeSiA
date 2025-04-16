
import numpy as np
import pandas as pd
import cv2
import torch
import torch.nn as nn
from pathlib import Path
from risk_estimation.datasets.frame_dropping import *
from risk_estimation.datasets.risk_dataloader import RiskEstimationDataset
from risk_estimation.datasets.risk_feature_extractor import *
from video_embedding.utils import get_session, load, save_models_index_list, save_video, save_video_index_list, tensor_image_to_cv2
from video_embedding.image_processing import saved_img_processing
import risk_estimation

def interp(original_array, new_length):
    new_indices = np.linspace(0, len(original_array) - 1, new_length)
    return np.interp(new_indices, np.arange(len(original_array)), original_array)

def test_interp():
    original_array = np.linspace(0, 10, 400)
    interp_arr = interp(original_array, new_length=350)

    assert len(interp_arr) == 350


def get_image_triplet(
        video_embedder, 
        images_numpy, 
        include_reconstruction_loss: bool = True, # Tested for True
        ): 
    images_cuda = torch.tensor(images_numpy, dtype=torch.float32).cuda()
    
    decoded_images = []
    original_images = []
    loss_title_images = []
    criterion = nn.MSELoss()
    cr = []
    for idx in range(len(images_numpy)):
        original_image = tensor_image_to_cv2(images_cuda[idx:idx+1])

        decoded_img1 = video_embedder.model.forward_batched(images_cuda[idx:idx+1])
        cr_ = criterion(decoded_img1, images_cuda[idx:idx+1])

        decoded_img = tensor_image_to_cv2(decoded_img1)
        cr.append(cr_)

        decoded_images.append(decoded_img)

        if include_reconstruction_loss:
            loss_title_image = np.zeros((16,64))
            cv2.putText(
                loss_title_image, 
                str(round(float(cr_),5)), 
                (0, 12), 
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 0, 0), 
                1, 
                2
            )
            loss_title_image = 255 - loss_title_image

        original_images.append(original_image)
        loss_title_images.append(loss_title_image)
    
    print("Averaged reconstruction loss: ", float(sum(cr)/len(cr)))
    
    image_triplets = []
    for ori, dec, sal in zip(original_images, decoded_images, loss_title_images):
        image_triplets.append(np.vstack((ori.squeeze(), dec.squeeze(), sal.squeeze())))

    image_triplets = np.array(image_triplets)

    return image_triplets

def video_triplets_save(video_name: str, video_embedder, folder="videos"):
    data = load(video_name)
    
    images_ = saved_img_processing(data['img']).squeeze().unsqueeze(1).cpu().numpy() # to format [images, 1, w, h]

    images = get_image_triplet(video_embedder, images_)

    video_name_without_trial = video_name.split("_trial_")[0]
    video_name_without_trial = video_name_without_trial.split("_test_")[0]

    # .. / video_skill / video_name with trial /
    path = f"{risk_estimation.path}/{folder}/{get_session()}/{video_name_without_trial}/{video_name}/"
    
    # make missing folders if missing
    Path(path).mkdir(parents=True, exist_ok=True)

    save_video(path, video_name, images, h=144, w=64)


def sample_and_save_on_video(video_name: str, video_embedder, risk_estimator, features, train_dataloader=None, folder="videos"):
    """ Test risk_estimator on sample video_name
        Save video (*.mp4) and risk series (*.csv) to folder (default: "/videos/")

    Args:
        video_name (str): Name
        video_embedder (_type_): _description_
        risk_estimator (_type_): _description_
        features (_type_): Feature Extraction
    """    
    if isinstance(features, str):
        features = eval(features)

    video_name_without_trial = video_name.split("_trial_")[0]
    video_name_without_trial = video_name_without_trial.split("_test_")[0]

    # .. / video_skill / video_name with trial /
    path = f"{risk_estimation.path}/{folder}/{get_session()}/{video_name_without_trial}/{video_name}/"
    
    # make missing folders if missing
    Path(path).mkdir(parents=True, exist_ok=True)


    dataset = RiskEstimationDataset.load_dataset([video_name], video_embedder,    
        frame_dropping_policy=NoFrameDroppingPolicy, # All frames are sampled 
        features=features
    )
    safe_labels = RiskEstimationDataset.load_dataset([video_name], video_embedder,    
        frame_dropping_policy=NoFrameDroppingPolicy, # All frames are sampled 
        features=LatentObservationsSafeLabels
    )

    pred, risks, std = risk_estimator.sample(dataset.X.squeeze())

    correct = (pred == dataset.Y.cpu().numpy().squeeze())
    safe_labels = safe_labels.Y.cpu().numpy().squeeze()
    risk_labels = dataset.Y.cpu().numpy().squeeze()

    has_label = np.ones(len(risks))

    df = pd.DataFrame(np.array([risks, correct, safe_labels, risk_labels, has_label, std]).T, columns=['Risk', 'Correct', 'SafeTrue', 'RiskTrue', 'HasLabel', 'Std'])
    df.to_csv(f"{path}/{video_name}_{risk_estimator.encode_params_as_str()}.csv", index_label='Time')

    save_models_index_list(path, video_name)

    save_video_index_list(f"{risk_estimation.path}/{folder}", parent_folder=folder)

