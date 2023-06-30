import os.path
import random

import numpy
import torch
from data.base_dataset import BaseDataset, get_params, get_transform_six_channel, get_transform_four
from data.image_folder import make_dataset
from PIL import Image
import cv2
import matplotlib.pyplot as plt


class ARIADataset(BaseDataset):
    """A dataset class for paired image dataset.

    It assumes that the directory '/path/to/data/train' contains image pairs in the form of {A,B}.
    During test time, you need to prepare a directory '/path/to/data/test'.
    """

    @staticmethod
    def modify_commandline_options(parser, is_train):
        # parser.add_argument('--fact_augment_size', type=int, default=64)
        parser.add_argument('--do_norm', type=bool, default=True)
        return parser

    def __init__(self, opt):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseDataset.__init__(self, opt)
        self.data_root = opt.dataroot

        assert (os.path.exists(self.data_root))

        self.len = len(os.listdir(os.path.join(self.data_root)))

        assert (self.opt.load_size >= self.opt.crop_size)  # crop_size should be smaller than the size of loaded image

        self.input_nc = 3
        self.output_nc = 1
        self.isTrain = opt.isTrain
        if self.isTrain:
            self.load_od = False
        else:
            self.load_od = opt.ignore_od

    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index - - a random integer for data indexing

        Returns a dictionary that contains A, B, A_paths and B_paths
            A (tensor) - - an image in the input domain
            B (tensor) - - its corresponding image in the target domain
            A_paths (str) - - image paths
            B_paths (str) - - image paths (same as A_paths)
        """

        original_path = os.path.join(self.data_root, str(index), 'image.png')
        label_path = os.path.join(self.data_root, str(index), 'label.png')
        mask_path = os.path.join(self.data_root, str(index), 'mask.png')
        if self.load_od:
            mask_od_path = os.path.join(self.data_root, str(index), 'mask_od.png')

        original = Image.open(original_path).convert('RGB')
        label = Image.open(label_path).convert('L')
        mask = Image.open(mask_path).convert('L')
        if self.load_od:
            mask_od = Image.open(mask_od_path).convert('L')

        transform_params = get_params(self.opt, original.size)
        raw_transform, label_transform = get_transform_six_channel(self.opt, transform_params, grayscale=False, do_norm=self.opt.do_norm)

        original = raw_transform(original)
        mask = label_transform(mask)
        label = label_transform(label)
        if self.load_od:
            mask_od = label_transform(mask_od)
            return {'image_original': original, 'mask': mask, 'label': label, 'source_path': original_path,
                    'mask_od': mask_od}

        return {'image_original': original, 'mask': mask, 'label': label, 'source_path': original_path}

    def __len__(self):
        """Return the total number of images in the dataset."""
        return self.len
