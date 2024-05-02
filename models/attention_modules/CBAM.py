from torch import nn
import torch


class CBAMChannelAttention(nn.Module):
    def __init__(self, in_channel, ratio=16):
        super(CBAMChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        # 这个MLP过程有点encoder-decoder的感觉，都是先压缩再解压
        self.fc1 = nn.Conv2d(in_channel, in_channel // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_channel // ratio, in_channel, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out  # [b, C, 1, 1]
        return self.sigmoid(out)


class CBAMSpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(CBAMSpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)  # 压缩通道
        max_out, _ = torch.max(x, dim=1, keepdim=True)  # 压缩通道
        x = torch.cat([avg_out, max_out], dim=1)  # [b, 1, h, w]
        x = self.conv1(x)
        return self.sigmoid(x)


class CABlock(nn.Module):
    def __init__(self, in_channel, do_residual=False, ca_ratio=16):
        super(CABlock, self).__init__()

        self.ca = CBAMChannelAttention(in_channel, ca_ratio)
        self.do_residual = do_residual

    def forward(self, x):
        out = self.ca(x) * x
        if self.do_residual:
            out = out + x
        return out


class SABlock(nn.Module):
    def __init__(self, in_channel, do_residual=False, sa_kernel_size=7):
        super(SABlock, self).__init__()

        self.sa = CBAMSpatialAttention(sa_kernel_size)
        self.do_residual = do_residual

    def forward(self, x):
        out = self.sa(x) * x
        if self.do_residual:
            out = out + x
        return out


class CBAMBlock(nn.Module):
    def __init__(self, in_channel, mode='ca_first', do_residual=False, ca_ratio=16, sa_kernel_size=7):
        super(CBAMBlock, self).__init__()

        self.ca = CBAMChannelAttention(in_channel, ca_ratio)
        self.sa = CBAMSpatialAttention(sa_kernel_size)
        self.mode = mode
        self.do_residual = do_residual

    def forward(self, x):
        if self.mode == 'ca_first':
            out = self.ca(x) * x
            out = self.sa(out) * out
        elif self.mode == 'sa_first':
            out = self.sa(x) * x
            out = self.ca(out) * out
        elif self.mode == 'parallel':
            ca_map = self.ca(x)
            sa_map = self.sa(x)
            out = (ca_map * x) * sa_map
        else:
            out = None

        if self.do_residual:
            out = out + x

        return out
