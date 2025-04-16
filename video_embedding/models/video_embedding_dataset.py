import torch


from torch.utils.data import DataLoader, random_split, TensorDataset
from video_embedding.utils import get_session, load
from video_embedding.image_processing import saved_img_processing

def load_dataloader(train_names, batch_size, shuffle=True):
    if isinstance(train_names, str):
        train_names = [train_names]

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

        next_image = resized_images.clone()
        # shift images to the left
        next_image[:-1] = resized_images[1:]
        next_image[-1] = resized_images[-1] 

        resized_images = torch.cat([resized_images, next_image], dim=1)

        image_tensor_list.append(resized_images)

    
    images = torch.cat(image_tensor_list, dim=0)
    dataset = TensorDataset(images)

    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
