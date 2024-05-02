import abc

import PIL
import numpy as np
import torch
from torchvision.transforms import transforms


class AugmentBase(abc.ABC):
    TWO_INPUTS = False
    REQUIRE_MASK = False

    @abc.abstractmethod
    def __call__(self, *args, **kwargs):
        pass

    @abc.abstractmethod
    def parallel_process(self, *args, **kwargs):
        pass

    @staticmethod
    def to_torch_tensor_and_add_batch(ori_image):
        if not isinstance(ori_image, torch.Tensor):
            if isinstance(ori_image, np.ndarray):
                ori_image = torch.FloatTensor(ori_image)
            elif isinstance(ori_image, PIL.Image.Image):
                ori_image = transforms.ToTensor()(ori_image) * 255
            else:
                raise TypeError('The input type of image is not supported.')
        else:
            ori_image = ori_image
        # check if the size of its shape is 4, if not add a batch channel
        if len(ori_image.shape) == 3:
            ori_image = ori_image.unsqueeze(0)
        return ori_image