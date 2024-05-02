import abc
import os
import random

import cv2
import numpy as np
import torch
from torch import nn
from scipy.special import comb

from utils.util import get_class_from_subclasses


class MaskGenerator(abc.ABC):

    @abc.abstractmethod
    def _generate_raw_mask(self, *args):
        pass

    def generate_mask(self, *args):
        mask_generated = self._generate_raw_mask(args)
        min_value, max_value = mask_generated.min(), mask_generated.max()
        mask_normalized = (mask_generated - min_value) / (max_value - min_value)
        return mask_normalized

    def image_fusion(self, image_1, image_2, *args):
        mask = self.generate_mask(args)
        return image_1 * mask + image_2 * (1 - mask)

    @classmethod
    def get_mask_generator_by_name(cls, generator_name='PatchMaskGenerator', *args):
        return get_class_from_subclasses(cls, generator_name)(*args)


# Gaussian Mixture Model
# too slow
class GMMMaskGenerator(MaskGenerator):
    def __init__(self, width=512, height=512, center_num=10, sigma_min=50, sigma_max=150):
        self.width = width
        self.height = height
        self.center_num = center_num
        self.sigma_range = (sigma_min, sigma_max)

    def _generate_raw_mask(self, *args):
        mu_x_list = [np.random.uniform(0, self.width) for _ in range(self.center_num)]
        mu_y_list = [np.random.uniform(0, self.height) for _ in range(self.center_num)]
        sigma_list = [np.random.uniform(*self.sigma_range) for _ in range(self.center_num)]
        mask_list = [np.zeros((self.width, self.height)) for _ in range(self.center_num)]
        for x in range(self.width):
            for y in range(self.height):
                for i in range(self.center_num):
                    mask_list[i][x, y] = np.exp(
                        - np.sqrt((mu_x_list[i] - x) ** 2 + (mu_y_list[i] - y) ** 2) / (2 * sigma_list[i])) / \
                                         sigma_list[i]
        mask_generated = np.zeros((self.width, self.height))
        for sub_mask in mask_list:
            temp_min, temp_max = sub_mask.min(), sub_mask.max()
            mask_generated += (sub_mask - temp_min) / (temp_max - temp_min)
        return mask_generated


# Distance Transformation
class DTMaskGenerator(MaskGenerator):
    def __init__(self, width=512, height=512, center_num=2):
        self.width = width
        self.height = height
        self.center_num = center_num

    def _generate_raw_mask(self, *args):
        mask_generated = np.ones((self.width, self.height), dtype=np.uint8)
        for _ in range(self.center_num):
            mask_generated[random.randint(0, 511), random.randint(0, 511)] = 0
        mask_generated = cv2.distanceTransform(mask_generated, cv2.DIST_L2, 5)
        return mask_generated


# expansion based
class DT2MASKGenerator(MaskGenerator):
    def __init__(self, width=512, height=512, center_num=5, expansion_step=100):
        self.width = width
        self.height = height
        self.center_num = center_num
        self.expansion_step = expansion_step
        self.max_pool = nn.MaxPool2d(kernel_size=5, stride=1, padding=2)

    def _generate_raw_mask(self, *args):
        mask_generated = torch.tensor(np.zeros((1, self.width, self.height), dtype=np.float32), requires_grad=False)
        for _ in range(self.center_num):
            mask_generated[0, random.randint(0, 511), random.randint(0, 511)] = 1
        final_mask = torch.tensor(np.zeros((1, self.width, self.height), dtype=np.float32), requires_grad=False)
        step_value = 1 / self.expansion_step
        for i in range(self.expansion_step):
            mask_generated = self.max_pool(mask_generated)
            final_mask += step_value * mask_generated
        return final_mask.squeeze().numpy()


# calculation per pixel
class DT3MaskGenerator(MaskGenerator):
    def __init__(self, width=512, height=512, center_num=3):
        self.width = width
        self.height = height
        self.center_num = center_num

    def _generate_raw_mask(self, *args):
        mask_generated = np.zeros((self.width, self.height), dtype=np.float32)
        center_x_list = [np.random.uniform(0, self.width) for _ in range(self.center_num)]
        center_y_list = [np.random.uniform(0, self.height) for _ in range(self.center_num)]
        pow_num = 2
        for i in range(self.width):
            for j in range(self.height):
                mask_generated[i, j] = min(
                    [pow(abs((x - i) ** pow_num) + abs((y - j) ** pow_num), 1 / pow_num) for x, y in
                     zip(center_x_list, center_y_list)])
        return mask_generated


class DT4MaskGenerator(MaskGenerator):
    def __init__(self, width=512, height=512, center_num=2):
        self.width = width
        self.height = height
        self.center_num = center_num

    def _generate_raw_mask(self, *args):
        mask_generated = np.zeros((self.width, self.height), dtype=np.float32)
        center_x_list = [np.random.uniform(0, self.width) for _ in range(self.center_num)]
        center_y_list = [np.random.uniform(0, self.height) for _ in range(self.center_num)]
        pow_num = 3
        for i in range(self.width):
            for j in range(self.height):
                temp = sum(
                    [abs((x - i) ** pow_num) + abs((y - j) ** pow_num) for x, y in zip(center_x_list, center_y_list)])
                mask_generated[i, j] = pow(temp, 1 / pow_num)
                # mask_generated[i, j] = min([pow(abs((x - i) ** pow_num + (y - j) ** pow_num), 1 / pow_num) for x, y in zip(center_x_list, center_y_list)])
        return mask_generated


def bezier_matrix(d):
    return np.array([[(-1) ** (i - j) * comb(j, i) * comb(d, j) for i in range(d + 1)] for j in range(d + 1)], int)


class BezierMaskGenerator(MaskGenerator):
    def __init__(self, width=512, height=512, grid_num=2):
        self.width = width
        self.height = height
        self.grid_num = grid_num

        u, v = np.arange(0, width), np.arange(0, height)
        u = np.array([u ** i for i in range(grid_num)])
        v = np.array([v ** i for i in range(grid_num)])
        bm_u, bm_v = bezier_matrix(grid_num - 1), bezier_matrix(grid_num - 1)
        self.m1 = u.T.dot(bm_u)
        self.m2 = bm_v.T.dot(v)

    def _generate_raw_mask(self, *args):
        grids = np.random.uniform(0, 1, (self.grid_num, self.grid_num))
        print(grids)
        return self.m1.dot(grids).dot(self.m2)


class GridMaskGenerator(MaskGenerator):
    def __init__(self, width=512, height=512, grid_num=4, reverse=False):
        width_list = (grid_num - 1) * [width // grid_num] + [width // grid_num + width % grid_num]
        height_list = (grid_num - 1) * [height // grid_num] + [height // grid_num + height % grid_num]
        sub_mask_list = []
        for i in range(grid_num):
            temp_list = []
            for j in range(grid_num):
                temp_list.append(np.zeros((width_list[i], height_list[j])) if ((i + j) % 2) ^ reverse else np.ones(
                    (width_list[i], height_list[j])))
            sub_mask_list.append(np.hstack(temp_list))
        self.raw_mask = np.vstack(sub_mask_list)

    def _generate_raw_mask(self, *args):
        return self.raw_mask.copy()


class PatchMaskGenerator(MaskGenerator):
    def __init__(self, width=512, height=512, side_length_range=None):
        self.width = width
        self.height = height
        if side_length_range is None:
            max_value = min(width, height)
            self.side_length_range = (max_value // 5, max_value // 2)

    def _generate_raw_mask(self, *args):
        mask_generated = np.zeros((self.width, self.height))
        patch_width = random.randint(*self.side_length_range)
        patch_height = random.randint(*self.side_length_range)
        patch_x = random.randint(0, self.width - patch_width)
        patch_y = random.randint(0, self.height - patch_height)
        mask_generated[patch_x: patch_x + patch_width, patch_y: patch_y + patch_height] = 1
        return mask_generated


class OfflineMaskGenerator(MaskGenerator):
    def __init__(self, width=512, height=512, mask_path=None, mask_type_list=('smooth',)):
        self.width = width
        self.height = height
        if mask_path is None:
            raise Exception('Please input a valid offline mask path')
        self.path_list = []
        # TODO: potential problem: the current implementation requires the same number of offline mask files for all three
        for mask_type in mask_type_list:
            for file_name in os.listdir(os.path.join(mask_path, mask_type)):
                self.path_list.append(os.path.join(mask_path, mask_type, file_name))

    def _generate_raw_mask(self, *args):
        offline_mask = cv2.imread(random.choice(self.path_list), cv2.IMREAD_GRAYSCALE) / 255
        if offline_mask.shape[0] != self.width or offline_mask.shape[1] != self.height:
            offline_mask = cv2.resize(offline_mask, (self.width, self.height))
        return offline_mask


if __name__ == '__main__':
    # mask = MaskGenerator.get_mask_generator_by_name('DT4MaskGenerator').generate_mask()
    # # print(mask_generated.shape)
    # import matplotlib.pyplot as plt
    # plt.imshow(mask)
    # plt.show()
    generator = OfflineMaskGenerator(mask_path='/data/lihaojin/random_mask')
    generator.generate_mask()
