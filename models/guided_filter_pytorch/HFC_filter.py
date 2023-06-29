# -*- coding: UTF-8 -*-
"""
@Function:
@File: HFC_filter.py
@Date: 2021/7/26 15:02 
@Author: Hever
"""
import os

from torch import nn
from torch.nn import functional as F
import torch
import cv2


class HFCFilter(nn.Module):
    def __init__(self,
                 # device,
                 filter_width=23, nsig=20, ratio=4, sub_low_ratio=1, sub_mask=False, is_clamp=True):
        super(HFCFilter, self).__init__()
        self.gaussian_filter = Gaussian_kernel(
            # device,
            filter_width, nsig=nsig)
        self.avg_pool = torch.nn.AvgPool2d(2, stride=2)
        self.max = 1.0
        self.min = -1.0
        self.ratio = ratio
        self.sub_low_ratio = sub_low_ratio
        self.sub_mask = sub_mask
        self.is_clamp = is_clamp

    def median_padding(self, x, mask):
        m_list = []
        batch_size = x.shape[0]
        for i in range(x.shape[1]):
            m_list.append(x[:, i].view([batch_size, -1]).median(dim=1).values.view(batch_size, -1) + 0.2)
        median_tensor = torch.cat(m_list, dim=1)
        median_tensor = median_tensor.unsqueeze(2).unsqueeze(2)
        mask_x = mask * x
        padding = (1 - mask) * median_tensor
        return padding + mask_x

    def forward(self, x, mask):
        assert mask is not None
        x = self.median_padding(x, mask)
        gaussian_output = self.gaussian_filter(x)
        res = self.ratio * (x - self.sub_low_ratio * gaussian_output)
        if self.is_clamp:
            res = torch.clamp(res, self.min, self.max)
        if self.sub_mask:
            res = (res + 1) * mask - 1

        return res


def get_kernel(kernel_len=16, nsig=10):  # nsig 标准差 ，kernlen=16核尺寸
    GaussianKernel = cv2.getGaussianKernel(kernel_len, nsig) \
                     * cv2.getGaussianKernel(kernel_len, nsig).T
    return GaussianKernel


class Gaussian_kernel(nn.Module):
    def __init__(self,
                 # device,
                 kernel_len, nsig=20):
        super(Gaussian_kernel, self).__init__()
        self.kernel_len = kernel_len
        kernel = get_kernel(kernel_len=kernel_len, nsig=nsig)  # 获得高斯卷积核
        kernel = torch.FloatTensor(kernel).unsqueeze(0).unsqueeze(0)  # 扩展两个维度
        # self.weight = nn.Parameter(data=kernel, requires_grad=False).to(device)
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


if __name__ == '__main__':
    from torchvision import transforms
    from PIL import Image, ImageDraw
    import random
    from tqdm import tqdm

    img = Image.open('/home/lihaojin/data/retinal_vessel/drive/train/0/image.png').convert('RGB')
    label = Image.open('/home/lihaojin/data/retinal_vessel/drive/train/0/label.png').convert('L')
    mask = Image.new("L", (512, 512), 0)
    draw = ImageDraw.Draw(mask)
    draw.ellipse(((0, 0), (511, 511)), fill=255)
    img = transforms.ToTensor()(img).unsqueeze(0)
    mask = transforms.ToTensor()(mask).unsqueeze(0)
    label = transforms.ToTensor()(label).unsqueeze(0)

    # 以下是生成HFC效果大图
    # # mid = (27, 9)
    # mid = (25, 11)
    # step = (4, 1)
    # amp = (6, 10)
    # width_list = list(range(mid[0] - amp[0] * step[0], mid[0] + amp[0] * step[0] + 1, step[0]))
    # sigma_list = list(range(mid[1] - amp[1] * step[1], mid[1] + amp[1] * step[1] + 1, step[1]))
    # print(width_list)
    # print(sigma_list)
    # hfc_list_list = [[HFCFilter(width, sigma, sub_low_ratio=1, sub_mask=True, is_clamp=True) for sigma in sigma_list] for width in width_list]
    # result_list = []
    #
    # for hfc_list in hfc_list_list:
    #     temp_list = []
    #     for hfc_filter in hfc_list:
    #         filtered = ((hfc_filter(img, mask) + 1) / 2).squeeze().numpy().transpose((1, 2, 0))[:, :, ::-1] * 255
    #         temp_list.append(filtered)
    #     result_list.append(cv2.hconcat(temp_list))
    # final_result = cv2.vconcat(result_list)
    #
    # cv2.imwrite('/home/lihaojin/final.png', final_result)

    # 以下是生成cutmix效果图
    target_dir = '/home/lihaojin/visual'
    if not os.path.exists(target_dir):
        os.mkdir(target_dir)
    cv2.imwrite(os.path.join(target_dir, 'image.png'), (img.squeeze().numpy().transpose((1, 2, 0)) * 255)[:, :, ::-1])
    cv2.imwrite(os.path.join(target_dir, 'mask.png'), mask.squeeze().numpy() * 255)
    cv2.imwrite(os.path.join(target_dir, 'label.png'), label.squeeze().numpy() * 255)

    h1 = HFCFilter(3, 3, sub_low_ratio=1, sub_mask=True, is_clamp=True)
    h2 = HFCFilter(5, 6, sub_low_ratio=1, sub_mask=True, is_clamp=True)
    h3 = HFCFilter(11, 9, sub_low_ratio=1, sub_mask=True, is_clamp=True)
    h4 = HFCFilter(21, 20, sub_low_ratio=1, sub_mask=True, is_clamp=True)
    h5 = HFCFilter(41, 30, sub_low_ratio=1, sub_mask=True, is_clamp=True)
    h6 = HFCFilter(81, 40, sub_low_ratio=1, sub_mask=True, is_clamp=True)
    h7 = HFCFilter(101, 80, sub_low_ratio=1, sub_mask=True, is_clamp=True)

    high1 = (h1(img, mask) + 1) / 2
    high2 = (h2(img, mask) + 1) / 2
    high3 = (h3(img, mask) + 1) / 2
    high4 = (h4(img, mask) + 1) / 2
    high5 = (h5(img, mask) + 1) / 2
    high6 = (h6(img, mask) + 1) / 2
    high7 = (h7(img, mask) + 1) / 2

    cv2.imwrite(os.path.join(target_dir, 'high1.png'), (high1.squeeze().numpy().transpose((1, 2, 0)) * 255)[:, :, ::-1])
    cv2.imwrite(os.path.join(target_dir, 'high2.png'), (high2.squeeze().numpy().transpose((1, 2, 0)) * 255)[:, :, ::-1])
    cv2.imwrite(os.path.join(target_dir, 'high3.png'), (high3.squeeze().numpy().transpose((1, 2, 0)) * 255)[:, :, ::-1])
    cv2.imwrite(os.path.join(target_dir, 'high4.png'), (high4.squeeze().numpy().transpose((1, 2, 0)) * 255)[:, :, ::-1])
    cv2.imwrite(os.path.join(target_dir, 'high5.png'), (high5.squeeze().numpy().transpose((1, 2, 0)) * 255)[:, :, ::-1])
    cv2.imwrite(os.path.join(target_dir, 'high6.png'), (high6.squeeze().numpy().transpose((1, 2, 0)) * 255)[:, :, ::-1])
    cv2.imwrite(os.path.join(target_dir, 'high7.png'), (high6.squeeze().numpy().transpose((1, 2, 0)) * 255)[:, :, ::-1])

    len_list = [50, 100, 150, 200, 250, 300, 350, 400]

    high_list = [high1, high2, high3, high4, high5, high6, high7]

    random.seed(0)

    for i in tqdm(range(7)):
        for j in range(7):
            for k in range(5):
                width = random.choice(len_list)
                height = random.choice(len_list)
                left = random.randint(0, 512 - width)
                up = random.randint(0, 512 - height)
                # print(left, up, width, height)
                result = high_list[i].clone()
                result[:, :, left:left + width, up:up + height] = high_list[j][:, :, left:left + width, up:up + height]
                cv2.imwrite(os.path.join(target_dir, 'mix_' + str(i + 1) + '_' + str(j + 1) + '_' + str(k) + '.png'),
                            (result.squeeze().numpy().transpose((1, 2, 0)) * 255)[:, :, ::-1])
