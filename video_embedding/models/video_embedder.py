import json
import pathlib
from typing import Iterable
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

import numpy as np
import cv2

import risk_estimation, video_embedding
# from video_embedding.models.elastic_weight_consolidation import ElasticWeightConsolidation
from video_embedding.utils import get_session, load
from video_embedding.models.nerual_networks.autoencoder import *
from tqdm import tqdm

from torch.utils.data import DataLoader

from video_embedding.image_processing import saved_img_processing

class VideoEmbedder(): #ElasticWeightConsolidation):
    def __init__(
        self,
        name: str,
        latent_dim: int = 12,
        batch_size: int = 40,
        frame_dropping=None,
        learning_rate: float = 0.01,
        nn_model: str = Autoencoder2,
    ):
        """Has scritly defined paths (see videos_path, models_path, latent_trajectory_path)
        Args:
            name (str): Skill and model name
            latent_dim (int, optional): Defaults to 8.
            batch_size (int, optional): Defaults to 40.
        """
        super(VideoEmbedder, self).__init__()
        self.name = name  # skill name
        self.model_train_record = []

        self.frame_dropping = frame_dropping
        
        if isinstance(nn_model, str):
            nn_model = eval(nn_model)
        self.model = nn_model(latent_dim)
        self.latent_dim = latent_dim
        # Move the model to GPU
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        # Define the loss function and optimizer
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)

        # Define batch size
        self.batch_size = batch_size  # Adjust the batch size as needed

    @property
    def videos_path(self):
        return f"{risk_estimation.path}/videos/{get_session()}/"

    @property
    def models_path(self):
        return f"{video_embedding.path}/saved_models/{get_session()}/"

    @property
    def latent_trajectory_path(self):
        return f"{video_embedding.path}/latent_trajectories/{get_session()}/"

    def save_latent_trajectory(self):
        path = self.latent_trajectory_path

        latent_traj = self.model.encoder(self.tensor_images)
        latent_traj = latent_traj.cpu().detach().numpy()
        pathlib.Path(path).mkdir(parents=True, exist_ok=True)
        np.savez(
            path + self.name + "_latent_" + str(self.latent_dim) + ".npz",
            latent_traj=latent_traj,
        )

    def load(self, videos: Iterable[str], shuffle=True):
        self.dataloader = self.load_dataset(videos)

    def load_dataset(
        self, train_names, shuffle=True
    ):
        if isinstance(train_names, str):
            train_names = [train_names]

        self.train_names = train_names

        image_tensor_list = []
        for n in range(len(train_names)):
            name = train_names[n]
            """ Loads data """
            data = load(file=name)
            images = data["img"]

            img_tensor = torch.tensor(images, dtype=torch.float32).cuda()
            resized_images = torch.zeros((len(images), 64, 64)).cuda()
            for i in range(len(images)):
                resized_images[i] = saved_img_processing(img_tensor[i])

            resized_images = resized_images.unsqueeze(1)  # Remove the channel dimension
            # resized_images = resized_images 

            image_tensor_list.append(resized_images)

        
        images = torch.cat(image_tensor_list, dim=0)
        dataset = TensorDataset(images)

        return DataLoader(dataset, batch_size=self.batch_size, shuffle=shuffle)

    def train(self, num_epochs: int):
        pviz = tqdm(range(num_epochs))
        try:
            for epoch in pviz:
                for data in self.dataloader:
                    input_batch = data[0]
                    # input_batch = data.flatten(-1)
                    self.optimizer.zero_grad()
                    output = self.model(input_batch)

                    loss = self.criterion(output, input_batch) # + self.ewc_loss()
                    loss.backward()
                    self.optimizer.step()

                pviz.set_description(
                    desc=f"Epoch [{epoch}/{num_epochs}], Loss: {loss.item()}"
                )
        except KeyboardInterrupt:
            pass
        # self.register_ewc_params()
        self.model_train_record.append(
            {
                "epoch": int(epoch),
                "loss": float(loss),
                "name": str(self.name),
                "train_names": list(self.train_names),
            }
        )
        return epoch, loss

    def create_video(self, dataloader: DataLoader = None):
        if dataloader is not None:
            self.dataloader = dataloader
        assert isinstance(
            self.dataloader, DataLoader
        ), f"Invalid dataloader: {self.dataloader}, {type(self.dataloader)}"
        
        self.model.eval()

        reconstructed_images = []
        for data in self.dataloader:
            output = self.model(data[0])
            reconstructed_images.append(output.cpu().detach().numpy())
        
        
        images_reconstruct = np.concatenate(reconstructed_images, axis=0).squeeze(1)
        # images_reconstruct = np.squeeze(output, axis=1)

        images_reconstruct = images_reconstruct * 255
        images_reconstruct = images_reconstruct.astype(np.uint8)


        pathlib.Path(self.videos_path).mkdir(parents=True, exist_ok=True)
        output_file = (
            self.videos_path
            + self.name
            + "_compressed_"
            + str(self.latent_dim)
            + ".avi"
        )
        frame_rate = 30


        # fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        fourcc = cv2.VideoWriter_fourcc(*"XVID")
        video_writer = cv2.VideoWriter(
            output_file, fourcc, frame_rate, (64, 64), isColor=False
        )

        # Write frames to the video file
        for i in range(len(images_reconstruct)):
            print("Writing frame:", i)
            # video_writer.write(images_reconstruct[i].reshape(64, 64).astype(np.uint8))
            video_writer.write(images_reconstruct[i])

        # Release the video writer and close the output file
        video_writer.release()

    def visulize_video(self, images):
        print("Lantent space projection")
        # Convert NumPy array to a PyTorch tensor

        for image in images:
            tensor_image = torch.tensor(image, dtype=torch.float32)
            tensor_image = tensor_image.cuda()
            tensor_image = tensor_image.unsqueeze(0).unsqueeze(0)
            latent_traj = self.model.encoder(tensor_image)

            print(latent_traj.cpu().detach().numpy())
            # image = (image * 255).astype(np.uint8)

            decoded_image = (
                self.model.forward(tensor_image).cpu().detach().numpy()[0][0]
            )
            image = (decoded_image).astype(np.uint8)
            cv2.imshow("Video", image)
            if cv2.waitKey(25) & 0xFF == 27:  # Press 'Esc' to exit
                break
        cv2.destroyAllWindows()

    def save_model(self):
        pathlib.Path(self.models_path).mkdir(
            parents=True, exist_ok=True
        )  # create dir if not exists
        with open(
            f"{self.models_path}/{self.name}_{self.model.__class__.__name__}_{self.latent_dim}.json", "w"
        ) as f:  # save config
            json.dump(self.model_train_record, f, indent=4)
        torch.save(
            self.model.state_dict(),
            f"{self.models_path}/{self.name}_{self.model.__class__.__name__}_{self.latent_dim}.pt",
        )  # save model

    def load_model(self):
        print(
            f"Loading model: {self.models_path}/{self.name}_{self.model.__class__.__name__}_{self.latent_dim}.pt"
        )

        state_dict = torch.load(
            f"{self.models_path}/{self.name}_{self.model.__class__.__name__}_{self.latent_dim}.pt"
        )

        # fisher and old params are saves as registered_buffer and not loaded as load_state_dict
        delete_params = []
        for name, param in state_dict.items():
            if "old_params_" in name or "fisher_" in name:  # is old param
                self.model.register_buffer(name, param)
                delete_params.append(name)

        self.model.load_state_dict(state_dict)
        self.model.eval()

        with open(
            f"{self.models_path}/{self.name}_{self.model.__class__.__name__}_{self.latent_dim}.json", "r"
        ) as f:
            model_train_record = json.load(f)
            if isinstance(model_train_record, dict):
                self.model_train_record = [model_train_record]
            else:
                self.model_train_record = model_train_record

