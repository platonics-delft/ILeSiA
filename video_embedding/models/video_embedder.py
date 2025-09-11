import json
import pathlib
from matplotlib import pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split

import numpy as np
import cv2

import risk_estimation, video_embedding
from video_embedding.utils import get_session
from video_embedding.models.nerual_networks.autoencoder import *
from video_embedding.models.nerual_networks.resnet_embedder import *
from tqdm import tqdm

class VideoEmbedder():
    def __init__(
        self,
        name: str,
        latent_dim: int = 12,
        learning_rate: float = 0.01,
        nn_model: str = Autoencoder3,
    ):
        """Has scritly defined paths (see videos_path, models_path, latent_trajectory_path)
        Args:
            name (str): Skill and model name
            latent_dim (int, optional):
        """
        super(VideoEmbedder, self).__init__()
        self.name = name  # skill name
        self.model_train_record = []
       
        if isinstance(nn_model, str):
            nn_model = eval(nn_model)
        self.model: nn.Module = nn_model(latent_dim)
        self.latent_dim = latent_dim
        # Move the model to GPU
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        # Define the loss function and optimizer
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(
            self.model.parameters(), lr=learning_rate, weight_decay=0.0000001
        )

    @property
    def videos_path(self):
        return f"{risk_estimation.path}/videos/{get_session()}/"

    @property
    def models_path(self):
        return f"{video_embedding.path}/saved_models/{get_session()}/"

    @property
    def latent_trajectory_path(self):
        return f"{video_embedding.path}/latent_trajectories/{get_session()}/"

    def save_latent_trajectory(self, tensor_images):
        path = self.latent_trajectory_path

        latent_traj = self.model.encoder(tensor_images)
        latent_traj = latent_traj.cpu().detach().numpy()
        pathlib.Path(path).mkdir(parents=True, exist_ok=True)
        np.savez(
            path + self.name + "_latent_" + str(self.latent_dim) + ".npz",
            latent_traj=latent_traj,
        )

    def nuclear_norm_loss(self, x):
        # Compute the nuclear norm of the input tensor
        # x = x.view(x.size(0), -1)  # Flatten the tensor
        u, s, v = torch.svd(x)
        return torch.sum(s)

    def split_dataloader(self, dataloader: DataLoader):
        """Splits the dataloader into two dataloaders"""
        dataset = dataloader.dataset
        dataset_size = len(dataset)
        train_size = int(np.floor(0.8 * dataset_size))
        test_size = dataset_size - train_size
        train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

        train_dataloader = DataLoader(
            train_dataset, batch_size=dataloader.batch_size, shuffle=True
        )
        test_dataloader = DataLoader(
            test_dataset, batch_size=dataloader.batch_size, shuffle=False
        )
        return train_dataloader, test_dataloader

    def train(self, dataloader, train_names, num_epochs: int = 1000):
        self.train_names = train_names
        try:
            self.training_loop(dataloader, num_epochs=num_epochs)

        except KeyboardInterrupt as e:
            print("Training interrupted by user.")
            self.save_model()
            raise e

    def training_loop(self, dataloader: DataLoader, num_epochs: int, patience = 100):    
        best_loss: float = float("inf")
        counter = 0

        train_loader, test_loader = self.split_dataloader(dataloader)

        pviz = tqdm(range(num_epochs))
        for epoch in pviz:
            if epoch % 10 == 0 and epoch > 0:
                self.optimizer.param_groups[0]["lr"] *= 0.5
            for data in train_loader:
                data_tensor = data[0]
                next_image = data_tensor[:, 1, :, :].unsqueeze(1)
                input_batch = data_tensor[:, 0, :, :].unsqueeze(1)
                self.optimizer.zero_grad()
                # output = self.model(input_batch)
                latent_vec = self.model.encoder(input_batch)
                output = self.model.decoder(latent_vec)

                latent_vec_next_image = self.model.encoder(next_image)

                alpha = 100.0
                beta = 0.0001
                l1_lambda = 0.00001
                continuity_loss_lambda = 1.0

                # continuity loss
                continuity_loss = F.mse_loss(latent_vec, latent_vec_next_image)

                # l1 loss on all weights
                l1_loss = 0
                for name, param in self.model.named_parameters():
                    l1_loss += torch.sum(torch.abs(param))

                # reconstruction loss
                recon_loss = self.criterion(output, input_batch)

                # nuclear norm loss
                nuclear_loss = self.nuclear_norm_loss(latent_vec)

                loss = (
                    alpha * recon_loss
                    + beta * nuclear_loss
                    + l1_lambda * l1_loss
                    + continuity_loss_lambda * continuity_loss
                )
                loss.backward()
                self.optimizer.step()

                train_loss = loss.item()

            # compute the test loss
            for data in test_loader:
                
                data_tensor = data[0]
                next_image = data_tensor[:, 1, :, :].unsqueeze(1)
                input_batch = data_tensor[:, 0, :, :].unsqueeze(1)
                # input_batch = data[0]
                with torch.no_grad():
                    latent_vec = self.model.encoder(input_batch)
                    output = self.model.decoder(latent_vec)
                    loss = self.criterion(output, input_batch)

            val_loss = loss.item()

            if val_loss < best_loss:
                best_loss = val_loss
                counter = 0  # Reset patience counter
            else:
                counter += 1

            if counter >= patience:  # Stop if no improvement for `patience` epochs
                print("Early stopping triggered")
                ret = "stop"
            else:
                ret = "continue"

            if ret == "stop":
                print(f"No improvement for {patience} epochs. Stopping training.")
                break

            pviz.set_description(
                desc=f"Epoch [{epoch}/{num_epochs}], Trainloss: {train_loss}, ValLoss: {val_loss}"
            )

        self.model_train_record.append(
            {
                "epoch": int(epoch),
                "loss": float(loss),
                "name": str(self.name),
                "train_names": list(self.train_names),
            }
        )
        return epoch, loss

    def latent_trajectory(self, dataloader: DataLoader = None):
        assert isinstance(dataloader, DataLoader), f"Invalid dataloader: {dataloader}, {type(dataloader)}"

        self.model.eval()

        latent_traj = []
        for data in dataloader:
            input_batch = data[0][:, 0, :, :].unsqueeze(1)
            output = self.model.encoder(input_batch)
            latent_traj.append(output.cpu().detach().numpy())

        latent_traj = np.concatenate(latent_traj, axis=0)
        return latent_traj

    def visualize_latent_trajectory(self, latent_traj, labels=None, perplexity=30):
        """Visualize the latent trajectory using t-SNE"""
        from sklearn.manifold import TSNE
        from sklearn.decomposition import PCA


        tsne = TSNE(n_components=2, perplexity=perplexity, max_iter=1000, random_state=42)
        latent_2d = tsne.fit_transform(latent_traj)

        pca = PCA(n_components=2)
        latent_2d_pca = pca.fit_transform(latent_traj)
        print(f"Explained variance ratio (PCA): {pca.explained_variance_ratio_}")
        print(f"Explained variance (PCA): {pca.explained_variance_}")
        print(f"Explained variance ratio (t-SNE): {tsne.kl_divergence_}")
        print(f"Explained variance (t-SNE): {tsne.kl_divergence_}")
        print(f"t-SNE perplexity: {perplexity}")
        print(f"t-SNE n_iter: {1000}")
        print(f"t-SNE random_state: {42}")

        # plot the pca trajectory
        plt.figure(figsize=(10, 8))
        plt.scatter(
            latent_2d_pca[:, 0],
            latent_2d_pca[:, 1],
            c=labels if labels is not None else "blue",
            cmap="viridis",
            alpha=0.7,
        )
        if labels is not None and len(np.unique(labels)) < 20:
            plt.colorbar(label="State/Pose Category")
        plt.title("PCA Visualization of Latent Space")
        plt.xlabel("Dimension 1")
        plt.ylabel("Dimension 2")
        plt.tight_layout()
        plt.savefig("pca_latent_space.png", dpi=300)

        # Plot the results
        plt.figure(figsize=(10, 8))
        scatter = plt.scatter(
            latent_2d[:, 0],
            latent_2d[:, 1],
            c=labels if labels is not None else "blue",
            cmap="viridis",
            alpha=0.7,
        )

        if labels is not None and len(np.unique(labels)) < 20:
            plt.colorbar(scatter, label="State/Pose Category")

        plt.title("t-SNE Visualization of Latent Space")
        plt.xlabel("Dimension 1")
        plt.ylabel("Dimension 2")
        plt.tight_layout()
        plt.savefig("tsne_latent_space.png", dpi=300)

    def create_video(self, dataloader: DataLoader = None):
        assert isinstance(dataloader, DataLoader), f"Invalid dataloader: {dataloader}, {type(dataloader)}"

        self.model.eval()

        reconstructed_images = []
        for data in dataloader:
            input_batch = data[0][:, 0, :, :].unsqueeze(1)
            output = self.model(input_batch)
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

    def load_model(self, path: pathlib.Path = None):
        if path is None:
            path = pathlib.Path(f"{self.models_path}/{self.name}_{self.model.__class__.__name__}_{self.latent_dim}.pt")
        else:
            assert path.exists(), f"Path {path} does not exist"
            assert path.is_file(), f"Path {path} is not a file"

        print(f"Loading model: {str(path)}")

        state_dict = torch.load(str(path))

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

