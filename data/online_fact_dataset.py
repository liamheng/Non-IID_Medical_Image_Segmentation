import os.path
import random

import numpy as np
import torch
from data.base_dataset import BaseDataset, get_params, get_transform_six_channel, get_transform_four
from data.image_folder import make_dataset
from PIL import Image
import cv2
import matplotlib.pyplot as plt
from utils import fda
from torchvision import transforms


class ONLINEFACTDataset(BaseDataset):
    """A dataset class for paired image dataset.

    It assumes that the directory '/path/to/data/train' contains image pairs in the form of {A,B}.
    During test time, you need to prepare a directory '/path/to/data/test'.
    """

    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser.add_argument('--target_root', type=str, default=None)
        parser.add_argument('--fact_l_lower', type=float, default=0.4)
        parser.add_argument('--fact_l_upper', type=float, default=0.8)
        parser.add_argument('--fact_mode', type=str, default='as')
        parser.add_argument('--do_target_norm', action='store_true')
        parser.add_argument('--maximum_data_num', required=False, type=int, default=None)
        return parser

    def __init__(self, opt):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseDataset.__init__(self, opt)
        self.data_root = opt.dataroot
        self.target_root = opt.target_root

        assert (os.path.exists(self.data_root))
        if self.target_root is not None:
            assert (os.path.exists(self.target_root))

        # self.len = len(os.listdir(os.path.join(self.data_root, 'original')))
        self.len = len(os.listdir(self.data_root))
        if opt.maximum_data_num is not None:
            if self.opt.maximum_data_num < self.opt.batch_size:
                self.len = opt.batch_size
            else:
                self.len = min(self.len, opt.maximum_data_num)

        if self.target_root is not None:
            self.target_list = os.listdir(self.target_root)

        assert (self.opt.load_size >= self.opt.crop_size)  # crop_size should be smaller than the size of loaded image

        self.input_nc = 3
        self.output_nc = 1
        self.isTrain = opt.isTrain

        self.fda_module = fda.FDAModule(opt.fact_mode)
        self.l_lower_bound = opt.fact_l_lower
        self.l_upper_bound = opt.fact_l_upper

        self.do_target_norm = opt.do_target_norm

        assert (0 <= self.l_lower_bound <= self.l_upper_bound <= 1)

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

        if self.opt.maximum_data_num is not None and self.opt.maximum_data_num < self.opt.batch_size:
            index = random.randint(0, self.opt.maximum_data_num - 1)

        image_path = os.path.join(self.data_root, str(index), 'image.png')
        label_path = os.path.join(self.data_root, str(index), 'label.png')
        mask_path = os.path.join(self.data_root, str(index), 'mask.png')
        if self.target_root is not None:
            target_path = os.path.join(self.target_root, random.choice(self.target_list))

        image = Image.open(image_path).convert('RGB')
        label = Image.open(label_path).convert('L')
        mask = Image.open(mask_path).convert('L') if os.path.exists(mask_path) else Image.new('L', label.size, 255)
        if self.target_root is not None:
            target = Image.open(target_path).convert('RGB')

        transform_params = get_params(self.opt, image.size)
        raw_transform, label_transform = get_transform_six_channel(self.opt, transform_params, grayscale=False,
                                                                   do_norm=False)

        image = raw_transform(image)
        mask = label_transform(mask)
        label = label_transform(label)
        if self.target_root is not None:
            target = transforms.ToTensor()(target)
            random_l = random.random() * (self.l_upper_bound - self.l_lower_bound) + self.l_lower_bound
            fact = self.fda_module(image, target, random_l)

        norm_func = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))

        image = norm_func(image)
        if self.target_root is not None:
            if self.opt.do_target_norm:
                target = norm_func(target)
            fact = norm_func(fact) * mask + mask - 1

        if self.opt.output_nc > 1:
            label = label * 255
            label = label.to(torch.long)

        if self.target_root is not None:
            return {'image_original': image, 'image_fact': fact, 'mask': mask,
                    'source_path': image_path, 'label': label, 'target': target}
        return {'image_original': image, 'mask': mask, 'source_path': image_path, 'label': label}

    def __len__(self):
        """Return the total number of images in the dataset."""
        return self.len
