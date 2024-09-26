
import os
import numpy as np
from video_embedding.utils import save_video
import torch

def test_video_save():

    tensor_images = torch.tensor(np.zeros((40,1,64,64))).cuda()

    save_video('', 'test_video', tensor_images)

    assert os.path.isfile("test_video.mp4")


if __name__ == '__main__':
    test_video_save()