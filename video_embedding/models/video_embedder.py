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
from video_embedding.models.elastic_weight_consolidation import ElasticWeightConsolidation
from video_embedding.utils import behaviour_trial_names, get_session, load, visualize_labelled_video
from video_embedding.models.nerual_networks.autoencoder import LargeAutoencoder, Autoencoder, CustomResnetStage1, CustomResnetStage2, CustomResnetStage3, CustomResnetStage4, CustomResnetStage5

import torchvision
from torchvision import transforms
from torchvision.transforms.functional import to_pil_image, to_tensor
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms.functional import adjust_brightness, adjust_contrast

VIEW_THE_AUGMENT = False

class CustomTensorDataset(Dataset):
    def __init__(self, tensors, transform=None):
        self.tensors = tensors
        self.transform = transform
    def __getitem__(self, index):
        x = self.tensors[index]
        if VIEW_THE_AUGMENT:
            visualize_labelled_video(x.cpu().numpy(), press_for_next_frame=True, printer=True)

        if self.transform:
            x_pil = to_pil_image(x)
            x_ = self.transform(x_pil)
            x_ = torch.tensor(256 * (1-x_), dtype=torch.float32).cuda()

        if VIEW_THE_AUGMENT:
            visualize_labelled_video(x_.cpu().numpy(), press_for_next_frame=True, printer=True)
        
        return x_
    def __len__(self):
        return len(self.tensors)

def random_brightness(img, max_delta=0.2):
    _factor = 1 + torch.randn(1).item() * max_delta
    return adjust_brightness(img, _factor)
def random_contrast(img, max_delta=0.2):
    _factor = 1 + torch.randn(1).item() * max_delta
    return adjust_contrast(img, _factor)

class VideoEmbedder(ElasticWeightConsolidation):
    def __init__(self, latent_dim=64, nn_model=Autoencoder, batch_size: int = 40, learning_rate: float = 0.01):
        super(VideoEmbedder, self).__init__()
        self.model = nn_model(latent_dim)
        self.latent_dim = latent_dim  
        # Move the model to GPU
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        # Define the loss function and optimizer
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)

        # Define batch size
        self.batch_size = batch_size # Adjust the batch size as needed
        if not hasattr(self, "augmentation"):
            self.augmentation = False

    def load(self, name: str):
        """Load videos dataset. Sets:
            self.dataloader has all train_names demonstrations 
            self.tensor_images has only images for "name" demonstration
        Args:
            name (str): Assigns name of the model
        """
        self.name = name
        self.dataloader = self.load_dataset([name])

    def load_dataset(self, train_names, validation=False):
        if not validation:
            self.train_names = train_names

        list_of_tensor_images = []
        for n in range(len(train_names)):
            name = train_names[n]
            ''' Loads data '''
            data = load(file=name)
            images= data['img']
            risk_flag = data['risk_flag']
            safe_flag = data['safe_flag']
            images_new=np.zeros((len(images),64,64))

            for i in range(len(images)):
                images_new[i]=cv2.resize(images[i], (64, 64))

            try:
                if self.frame_dropping == True: # Is False by default
                    images_new = self.only_labelled_frame_drop(images_new, risk_flag, safe_flag)
            except AttributeError:
                pass

            images = images_new[:, np.newaxis, :, :]
            # Assuming 'images' is your NumPy array of shape (num_images, h, w)

            # Convert NumPy array to a PyTorch tensor
            tensor_images = torch.tensor(images, dtype=torch.float32).cuda()

            if n == 0:
                if not validation:
                    self.tensor_images = tensor_images

            list_of_tensor_images.append(tensor_images)

        tensor_dataset_images = torch.cat(list_of_tensor_images)
        # Create a TensorDataset
        if self.augmentation:
            transform = transforms.Compose([
                transforms.Grayscale(num_output_channels=1),
                transforms.RandomPerspective(distortion_scale=0.1, p=0.5,interpolation=2),
                transforms.RandomAffine(degrees=0, translate=(0.015,0.015)),
                transforms.Lambda(lambda img: random_brightness(img, 0.1)),
                transforms.Lambda(lambda img: random_contrast(img, 0.1)),
                transforms.ToTensor(),
                
                ])
            dataset = CustomTensorDataset(tensor_dataset_images, transform=transform)
        else:
            dataset = TensorDataset(tensor_dataset_images)

        return DataLoader(dataset, batch_size=self.batch_size, shuffle=False)


    def only_labelled_frame_drop(self, images, risk_flags, safe_flags):
        risk_flags = risk_flags.squeeze()
        safe_flags = safe_flags.squeeze()
        images_labelled = []
        for i in range(len(images)):
            if risk_flags[i] or safe_flags[i]:
                images_labelled.append(images[i])
        print(f"Number of images {len(images_labelled)}")
        
        return np.array(images_labelled)


    # Train the autoencoder
    def training_loop(self, num_epochs: int, patience: int):
        
        stopping_logic = EarlyStopping(patience)
        for epoch in range(num_epochs):
            for data in self.dataloader:
                if self.augmentation:
                    input_batch = data.flatten(-1)
                else:
                    input_batch = data[0]
                self.optimizer.zero_grad()
                output = self.model(input_batch)
                
                loss = self.ewc_loss() + self.criterion(output, input_batch)
                loss.backward()
                self.optimizer.step()
                
                if not self.augmentation:
                    for valid_data in self.validation_dataloader:
                        valid_batch = valid_data[0]
                        break
                    valid_output = self.model(valid_batch)
                    valid_loss = self.ewc_loss() + self.criterion(valid_output, valid_batch)
                    valid_loss_fp = float(valid_loss.item())
                    del valid_data
                    del valid_batch
                    del valid_output
                    del valid_loss
                
            if self.augmentation:
                ret = stopping_logic(loss) #valid_loss_fp)
            else:
                ret = stopping_logic(valid_loss_fp)
            if ret == 'stop':
                print(f"No improvement for {patience} epochs. Stopping training.")
                break

            if epoch % 5== 0:
                if self.augmentation:
                    print('Epoch [{}/{}], Loss: {:.4f}, Valid. Loss: ?'.format(epoch+1, num_epochs, loss.item() )) #, valid_loss_fp))
                else:
                    print('Epoch [{}/{}], Loss: {:.4f}, Valid. Loss: {:.4f}'.format(epoch+1, num_epochs, loss.item(), valid_loss_fp))


        if not self.augmentation:
            self.register_ewc_params()        
        return epoch, loss

    def create_video(self, epoch: int = 1000, patience: int = 1000):
        # train autoencoder
        self.training_loop(num_epochs = epoch, patience = patience)
        # feedforward, reconstruct image
        output = self.model(self.tensor_images)
        output = output.cpu().detach().numpy()

        latent_traj = self.model.encoder(self.tensor_images)
        latent_traj = latent_traj.cpu().detach().numpy()

        images_reconstruct= np.squeeze(output, axis=1)

        pathlib.Path(self.videos_path).mkdir(parents=True, exist_ok=True)
        output_file = self.videos_path + self.name +'_compressed_'+ str(self.latent_dim) +'.avi'
        frame_rate = 30
        # fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        video_writer = cv2.VideoWriter(output_file, fourcc, frame_rate, (64, 64), isColor=False)

        # Write frames to the video file
        for i in range(len(images_reconstruct)):
            print("Writing frame:",i)
            video_writer.write(images_reconstruct[i].reshape(64, 64).astype(np.uint8))
            

        # Release the video writer and close the output file
        video_writer.release()


    def visulize_video(self, images):
        print("Lantent space projection")
        # Convert NumPy array to a PyTorch tensor

        for image in images:
            tensor_image = torch.tensor(image, dtype=torch.float32)
            tensor_image= tensor_image.cuda()
            tensor_image=tensor_image.unsqueeze(0).unsqueeze(0)
            latent_traj = self.model.encoder(tensor_image)

            print(latent_traj.cpu().detach().numpy())
            # image = (image * 255).astype(np.uint8)

            decoded_image = self.model.forward(tensor_image).cpu().detach().numpy()[0][0]
            image = (decoded_image).astype(np.uint8)
            cv2.imshow('Video',image)
            if cv2.waitKey(25) & 0xFF == 27:  # Press 'Esc' to exit
                break
        cv2.destroyAllWindows()

    def save_model(self, path=None):
        torch.save(self.model.state_dict(), f"{path}/{self.name}_model_{self.latent_dim}.pt")

    def load_model(self, path=None, name=None, latent_dim=None):
        if name is not None: self.name = name
        if latent_dim is not None: self.latent_dim = latent_dim
        print(f"Loaded model: {path}/{self.name}_model_{self.latent_dim}.pt")
        self.model.load_state_dict(torch.load(f"{path}/{self.name}_model_{self.latent_dim}.pt"))
        self.model.eval()
    
    def save_latent_trajectory(self, path=None):
        latent_traj = self.model.encoder(self.tensor_images)
        latent_traj = latent_traj.cpu().detach().numpy()
        pathlib.Path(path).mkdir(parents=True, exist_ok=True)
        np.savez(path +self.name +'_latent_'+ str(self.latent_dim) +'.npz', latent_traj=latent_traj)


class RiskyBehavioralVideoEmbedder(VideoEmbedder):
    def __init__(self,
                name: str,
                latent_dim: int = 8,
                batch_size: int = 40,
                behaviours: Iterable[str] = None,
                frame_dropping=None,
                learning_rate: float = 0.01,
                augmentation: bool = True,
                nn_model: str = Autoencoder,
                ):
        """Has scritly defined paths (see videos_path, models_path, latent_trajectory_path)
            When behaviours given: Model is saved to subfolder named "behaviour1_behaviours2_behaviour3_..."
        
        Args:
            name (str): Skill and model name
            latent_dim (int, optional): Defaults to 8.
            batch_size (int, optional): Defaults to 40.
            behaviours (Iterable[str], optional): Defaults to None.
                (see trajectory_data/trajectories/<demonstration>_description.csv)
            augmentation (bool, optional): Apply affine, perspective, brightness, contrast transformations
        """        
        if isinstance(nn_model, str):
            nn_model = eval(nn_model)
        
        super().__init__(latent_dim=latent_dim, nn_model=nn_model, batch_size=batch_size, learning_rate=learning_rate)
        self.name = name # skill name
        self.behaviours = behaviours
        self.model_train_record = []
        
        self.frame_dropping = frame_dropping
        self.augmentation = augmentation

    def training_loop(self, num_epochs: int, patience: int):
        epoch, loss = super().training_loop(num_epochs, patience)
        self.model_train_record.append({
            "epoch": int(epoch),
            "loss": float(loss),
            "name": str(self.name),
            "behaviours": self.behaviours,
            "train_names": list(self.train_names),
        })

    @property
    def videos_path(self):
        return f"{risk_estimation.path}/videos/{get_session()}/"

    @property
    def models_path(self):
        return f"{video_embedding.path}/saved_models/{get_session()}/{self.encode_behaviour_folder()}"

    @property
    def latent_trajectory_path(self):
        return f"{video_embedding.path}/latent_trajectories/{get_session()}/"

    def save_latent_trajectory(self):
        return super().save_latent_trajectory(self.latent_trajectory_path)

    def encode_behaviour_folder(self):
        # Behaviour names to folder name
        # ["successful", "hand", "cables"] -> "successful_hand_cables"
        if self.behaviours is None:
            behaviour_folder = ""
        else:
            behaviour_folder = "_".join(self.behaviours.copy())
        return behaviour_folder

    def load(self, videos: Iterable[str], validation_videos: Iterable[str] = []):
        """Public interface to load videos. The 'videos' can be a string or a list of strings.
        Args:
            videos (str | Iterable[str]) 
        """
        if isinstance(validation_videos, Iterable) and len(validation_videos) == 0:
            vdl = []
        elif isinstance(validation_videos, str):
            if self.behaviours is None:
                vdl = self._load_single_video(validation_videos, validation=True)
            elif isinstance(self.behaviours, Iterable):
                vdl = self._load_single_video_behaviours(validation_videos, self.behaviours, validation=True)
        elif isinstance(validation_videos, Iterable):
            if self.behaviours is None:
                vdl = self._load_multi_videos(validation_videos, validation=True)
            elif isinstance(self.behaviours, Iterable):
                vdl = self._load_multi_video_behaviours(validation_videos, self.behaviours, validation=True)
        assert 'vdl' in locals(), f"Invalid argument validation_videos: {validation_videos}, {type(validation_videos)}, {isinstance(validation_videos, Iterable)}, {len(validation_videos) == 0}"

        self.validation_dataloader = vdl

        if isinstance(videos, str):
            if self.behaviours is None:
                dl = self._load_single_video(videos)
            elif isinstance(self.behaviours, Iterable):
                dl = self._load_single_video_behaviours(videos, self.behaviours)
        elif isinstance(videos, Iterable):
            if self.behaviours is None:
                dl = self._load_multi_videos(videos)
            elif isinstance(self.behaviours, Iterable):
                dl = self._load_multi_video_behaviours(videos, self.behaviours)
        assert dl, "Invalid argument videos"
    
        self.dataloader = dl



    def _load_single_video(self, video, validation=False):
        print(f"Training on single video: {video}")
        return self.load_dataset([video], validation=validation)

    def _load_multi_videos(self, videos, validation=False):
        print(f"Training on multiple videos: {videos}")
        return self.load_dataset(videos, validation=validation)

    def _load_single_video_behaviours(self, video, behaviours, validation=False):
        # Loads listed videos in trajectory_data/trajectories/<demonstration>_description.csv file
        train_names = behaviour_trial_names(self.name, behaviours)
        
        print(f"Training single video {video} with behaviours {behaviours}")
        print(f"All video names: {train_names}")
        return self.load_dataset(train_names, validation=validation)

    def _load_multi_video_behaviours(self, videos, behaviours, validation=False):
        # For every behaviour, loads listed videos in trajectory_data/trajectories/<demonstration>_description.csv file
        train_names = list(np.array([behaviour_trial_names(name, behaviours) for name in videos]).flatten())
        
        print(f"Training multiple videos {videos} with behaviours {behaviours}")
        print(f"All video names: {train_names}")
        return self.load_dataset(train_names, validation=validation)

    def save_model(self):
        pathlib.Path(self.models_path).mkdir(parents=True, exist_ok=True) # create dir if not exists
        with open(f"{self.models_path}/{self.name}_model_{self.latent_dim}.json", 'w') as f: # save config
            json.dump(self.model_train_record, f, indent=4)
        torch.save(self.model.state_dict(), f"{self.models_path}/{self.name}_model_{self.latent_dim}.pt") # save model

    def load_model(self):
        print(f"Loading model: {self.models_path}/{self.name}_model_{self.latent_dim}.pt")

        state_dict = torch.load(f"{self.models_path}/{self.name}_model_{self.latent_dim}.pt")

        # fisher and old params are saves as registered_buffer and not loaded as load_state_dict
        delete_params = []
        for name, param in state_dict.items():
            if 'old_params_' in name or 'fisher_' in name: # is old param
                self.model.register_buffer(name, param)
                delete_params.append(name)

        self.model.load_state_dict(state_dict)
        self.model.eval()

        with open(f"{self.models_path}/{self.name}_model_{self.latent_dim}.json", 'r') as f:
            model_train_record = json.load(f)
            if isinstance(model_train_record, dict):
                self.model_train_record = [model_train_record]
            else:
                self.model_train_record = model_train_record


class EarlyStopping:
    def __init__(self, patience=100, delta=0):
        self.patience = patience
        self.delta = delta
        self.best_score = None
        self.epochs_no_improve = 0

        self.val_loss_min = np.Inf
        
        self.smoothed_loss = None
        
        self.alpha = 0.1  # Smoothing factor

    def __call__(self, val_loss):
        if self.smoothed_loss is None:
            self.smoothed_loss = val_loss
        else:
            self.smoothed_loss = self.alpha * val_loss + (1 - self.alpha) * self.smoothed_loss

        score = -self.smoothed_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss)
        elif score < self.best_score + self.delta:
            self.epochs_no_improve += 1
            if self.epochs_no_improve >= self.patience:
                return "stop"
        else:
            self.best_score = score
            self.save_checkpoint(val_loss)
            self.epochs_no_improve = 0
        return "continue"

    def save_checkpoint(self, val_loss):
        '''Saves model when validation loss decrease.'''
        if val_loss < self.val_loss_min:
            self.val_loss_min = val_loss