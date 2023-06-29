from models.guided_filter_pytorch.HFC_filter import *
import random


class GaussianMixUp(nn.Module):

    def __init__(self, width_list=tuple(range(5, 50, 2)), sigma_list=tuple(range(2, 22)), mixup_size=50,
                 sub_low_ratio=1, sub_mask=True, is_clamp=True):
        super(GaussianMixUp, self).__init__()

        self.hfc_list = nn.ModuleList(
            [HFCFilter(width, sigma, sub_low_ratio=sub_low_ratio, sub_mask=sub_mask, is_clamp=is_clamp) for width, sigma
             in zip(width_list, sigma_list)])
        self.mixup_size = mixup_size

    def forward(self, image, mask):
        *_, w, h = image.shape

        if self.mixup_size == -1:
            mixup_width = random.randrange(0, w)  # todo 确认这个范围是否正确
            mixup_height = random.randrange(0, h)
        elif self.mixup_size < -1:
            mixup_width = random.randrange(0, -self.mixup_size + 1)
            mixup_height = -self.mixup_size - mixup_width
        else:
            mixup_width = self.mixup_size
            mixup_height = self.mixup_size

        pos_x = random.randrange(0, w - mixup_width)
        pos_y = random.randrange(0, h - mixup_height)

        hfc_filter_1 = random.choice(self.hfc_list)
        hfc_filter_2 = random.choice(self.hfc_list)

        result_1 = hfc_filter_1(image, mask)
        result_2 = hfc_filter_2(image, mask)

        result_1[:, :, pos_x:pos_x + mixup_width, pos_y:pos_y + mixup_height] = \
            result_2[:, :, pos_x:pos_x + mixup_width, pos_y:pos_y + mixup_height]

        return result_1
