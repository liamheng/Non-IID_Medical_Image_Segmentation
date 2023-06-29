import abc
from enum import Enum, unique

import numpy as np
import torch
import cv2

from torch import nn
import torch.nn.functional as F


class HFCFilter(nn.Module, abc.ABC):
    def __init__(self, do_median_padding=True, normalization_percentile_threshold=3, sub_mask=True):
        super(HFCFilter, self).__init__()
        self.do_median_padding = do_median_padding
        self.normalization_percentile_threshold = normalization_percentile_threshold
        self.sub_mask = sub_mask

    @staticmethod
    def median_padding(x, mask):
        m_list = []
        batch_size = x.shape[0]
        for i in range(x.shape[1]):
            m_list.append(x[:, i].view([batch_size, -1]).median(dim=1).values.view(batch_size, -1) + 0.2)
        median_tensor = torch.cat(m_list, dim=1)
        median_tensor = median_tensor.unsqueeze(2).unsqueeze(2)
        mask_x = mask * x
        padding = (1 - mask) * median_tensor
        return padding + mask_x

    @abc.abstractmethod
    def get_hfc(self, x, mask):
        pass

    def forward(self, x, mask):
        if self.do_median_padding:
            x = self.median_padding(x, mask)
        res = self.get_hfc(x, mask)
        # do normalization
        N, C = res.shape[0], res.shape[1]
        for n in range(N):
            for c in range(C):
                temp_res = (res * 256).int().float() / 256
                res_min, res_max = np.percentile(temp_res[n, c].detach().cpu().numpy(),
                                                 self.normalization_percentile_threshold), np.percentile(
                    temp_res[n, c].detach().cpu().numpy(), 100 - self.normalization_percentile_threshold)
                res[n, c] = (res[n, c] - res_min) / (res_max - res_min)
        if self.sub_mask:
            res = res * mask
        return res


class GaussianKernel(nn.Module):
    def __init__(self,
                 # device,
                 kernel_len, nsig=20):
        super(GaussianKernel, self).__init__()
        self.kernel_len = kernel_len
        kernel = cv2.getGaussianKernel(kernel_len, nsig) * cv2.getGaussianKernel(kernel_len, nsig).T
        kernel = torch.FloatTensor(kernel).unsqueeze(0).unsqueeze(0)  # 扩展两个维度
        self.weight = nn.Parameter(data=kernel, requires_grad=False)

        self.padding = torch.nn.ReplicationPad2d(int(self.kernel_len / 2))

    def forward(self, x):  # x1是用来计算attention的，x2是用来计算的Cs
        x = self.padding(x)
        # 对三个channel分别做卷积
        res = []
        for i in range(x.shape[1]):
            res.append(F.conv2d(x[:, i:i + 1], self.weight))
        x_output = torch.cat(res, dim=1)
        return x_output


class GaussianHFCFilter(HFCFilter):
    def __init__(self, filter_width=23, nsig=9, ratio=4, sub_low_ratio=1, do_median_padding=True,
                 normalization_percentile_threshold=3, sub_mask=True):
        super(GaussianHFCFilter, self).__init__(do_median_padding=do_median_padding,
                                                normalization_percentile_threshold=normalization_percentile_threshold,
                                                sub_mask=sub_mask)
        self.gaussian_filter = GaussianKernel(filter_width, nsig=nsig)
        self.ratio = ratio
        self.sub_low_ratio = sub_low_ratio

    def get_hfc(self, x, mask):
        gaussian_output = self.gaussian_filter(x)
        return self.ratio * (x - self.sub_low_ratio * gaussian_output)


class FourierButterworthHFCFilter(HFCFilter):
    def __init__(self, image_size=(512, 512), butterworth_d0_ratio=0.05, butterworth_n=1,
                 do_median_padding=True, normalization_percentile_threshold=3, sub_mask=True):
        super(FourierButterworthHFCFilter, self).__init__(do_median_padding=do_median_padding,
                                                          normalization_percentile_threshold=normalization_percentile_threshold,
                                                          sub_mask=sub_mask)
        self.image_size = image_size

        # distance transform
        d0 = int(max((butterworth_d0_ratio * min(*image_size)) // 2, 1))
        corners = ((0, 0), (0, image_size[1] - 1), (image_size[0] - 1, image_size[1] - 1), (image_size[0] - 1, 0))
        self.filter_map = np.zeros(image_size, dtype=np.float32)
        for i in range(image_size[0]):
            for j in range(image_size[1]):
                d = np.inf
                for x, y in corners:
                    d = min(d, np.sqrt((i - x) ** 2 + (j - y) ** 2))
                self.filter_map[j, i] = 1 / (1 + (d0 / (d + 1)) ** (2 * butterworth_n))
        self.filter_map = nn.Parameter(data=torch.FloatTensor(self.filter_map), requires_grad=False)

    def get_hfc(self, x, mask):
        x_fft = torch.fft.fft2(x)
        x_fft_temp = x_fft * self.filter_map
        return torch.abs(torch.fft.ifft2(x_fft_temp))


class FourierStepHFCFilter(HFCFilter):
    def __init__(self, image_size=(512, 512), step_ratio=0.5,
                 do_median_padding=True, normalization_percentile_threshold=3, sub_mask=True):
        super(FourierStepHFCFilter, self).__init__(do_median_padding=do_median_padding,
                                                   normalization_percentile_threshold=normalization_percentile_threshold,
                                                   sub_mask=sub_mask)
        self.image_size = image_size

        # distance transform
        d0 = int(max((step_ratio * min(*image_size)) // 2, 1))
        corners = ((0, 0), (0, image_size[1] - 1), (image_size[0] - 1, image_size[1] - 1), (image_size[0] - 1, 0))
        self.filter_map = np.zeros(image_size, dtype=np.float32)
        self.filter_map = cv2.rectangle(self.filter_map, (0, 0), (d0, d0), 1, -1)  # up left
        self.filter_map = cv2.rectangle(self.filter_map, (0, image_size[1] - d0), (d0, image_size[1]), 1,
                                        -1)  # down left
        self.filter_map = cv2.rectangle(self.filter_map, (image_size[0] - d0, 0), (image_size[0], d0), 1,
                                        -1)  # up right
        self.filter_map = cv2.rectangle(self.filter_map, (image_size[0] - d0, image_size[1] - d0),
                                        (image_size[0], image_size[1]), 1, -1)  # down right
        self.filter_map = nn.Parameter(data=torch.FloatTensor(self.filter_map), requires_grad=False)

    def get_hfc(self, x, mask):
        x_fft = torch.fft.fft2(x)
        x_fft_temp = x_fft * self.filter_map
        return torch.abs(torch.fft.ifft2(x_fft_temp))
