import random

import numpy as np
import torch
from torch import nn

from models.guided_filter_pytorch.ModifiedHFCFilter import FourierButterworthHFCFilter, FourierStepHFCFilter, \
    IdentityHFCFilter
from .augment_prototype import AugmentBase
from utils.mixup_mask import OfflineMaskGenerator


class FMAug(AugmentBase):
    TWO_INPUTS = False
    REQUIRE_MASK = True

    def __init__(self, mixup_type='patch,grid,smooth', hfc_type='butterworth_all_channels', remain_original='False', device='cuda'):
        self.device = torch.device(device)
        torch.multiprocessing.set_start_method('spawn')
        self.mixup_type = mixup_type
        if mixup_type != 'None':
            self.mask_generator = OfflineMaskGenerator(mask_path='/data/lihaojin/random_mask',
                                                       mask_type_list=mixup_type.split(','))

        self.hfc_type = hfc_type
        self.hfc_list = nn.ModuleList()
        if hfc_type == 'butterworth_all_channels' or hfc_type == 'butterworth_per_channel':
            d0_num = 12
            d0_max = 0.1
            remap_ratio = 4.1
            n_list = [1, 2, 3]
            d0_list = d0_max * (np.exp(remap_ratio * np.linspace(0, d0_max, d0_num) / d0_max) - 1) / (
                    np.exp(remap_ratio) - 1)
            for i in range(len(n_list) * d0_num):
                self.hfc_list.append(
                    FourierButterworthHFCFilter(butterworth_d0_ratio=d0_list[i // 3],
                                                butterworth_n=n_list[i % len(n_list)],
                                                do_median_padding=False, image_size=(512, 512)).to(device=self.device))
        elif hfc_type == 'step':
            step_granularity = 20
            for i in range(step_granularity):
                self.hfc_list.append(
                    FourierStepHFCFilter(image_size=(512, 512), step_ratio=i / step_granularity).to(device=self.device))

        if remain_original == 'True':
            # 1/4 identity
            self.hfc_list += [IdentityHFCFilter().to(device=self.device)] * int(len(self.hfc_list) / 3)

    def do_hfc(self, img, mask):
        if self.hfc_type == 'butterworth_per_channel':
            return torch.cat([random.choice(self.hfc_list)(img[:, i].unsqueeze(1), mask) for i in range(3)], dim=1)
        else:
            return random.choice(self.hfc_list)(img, mask)

    def __call__(self, image, mask):
        image = self.to_torch_tensor_and_add_batch(image).to(device=self.device)
        mask = self.to_torch_tensor_and_add_batch(mask).to(device=self.device)
        image_1 = self.do_hfc(image, mask).detach().cpu().numpy()
        if self.mixup_type == 'None':
            return image_1.transpose(0, 2, 3, 1)[0] if image_1.shape[0] == 1 else image_1.transpose(0, 2, 3, 1)
        image_2 = self.do_hfc(image, mask).detach().cpu().numpy()
        merge_mask = self.mask_generator.generate_mask()[np.newaxis, np.newaxis, :, :, ]
        image_output = merge_mask * image_1 + (1 - merge_mask) * image_2
        return image_output.transpose(0, 2, 3, 1)[0] if image_output.shape[0] == 1 else image_output.transpose(0, 2, 3, 1)

    def parallel_process(self, image):
        if not isinstance(image, np.ndarray):
            image = np.asarray(image)
        return image.copy()
