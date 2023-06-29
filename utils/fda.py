import torch
from torch import nn
import numpy as np

# pytorch版本问题
try:
    from torch import irfft
    from torch import rfft
except ImportError:
    from torch.fft import irfft2
    from torch.fft import rfft2


    def rfft(x, d):
        t = rfft2(x, dim=(-d))
        return torch.stack((t.real, t.imag), -1)


    def irfft(x, d, signal_sizes):
        return irfft2(torch.complex(x[:, :, 0], x[:, :, 1]), s=signal_sizes, dim=(-d))


# FDA
def mutate_as(amp_src, amp_trg, L=0.1):
    *_, h, w = amp_src.size()
    b = (np.floor(np.amin((h, w)) * L)).astype(int)  # get b
    amp_src[:, h // 2 - b:h // 2 + b, w // 2 - b:w // 2 + b] = \
        amp_trg[:, h // 2 - b:h // 2 + b, w // 2 - b:w // 2 + b]
    return amp_src


# FACT
def mutate_am(amp_src, amp_trg, L=0.1):
    amp_src = (1 - L) * amp_src + L * amp_trg
    return amp_src


def mutate_inter(amp_src, amp_trg, L=(0.05, 0.5)):
    if isinstance(L, int) or isinstance(L, float):
        L = (0.05, L)
    *_, h, w = amp_src.size()
    b = (np.floor(np.amin((h, w)) * L[0])).astype(int)  # get b
    amp_src[:, h // 2 - b:h // 2 + b, w // 2 - b:w // 2 + b] = \
        L[1] * amp_src[:, h // 2 - b:h // 2 + b, w // 2 - b:w // 2 + b] + \
        (1 - L[1]) * amp_trg[:, h // 2 - b:h // 2 + b, w // 2 - b:w // 2 + b]
    return amp_src


func_dict = {'as': mutate_as, 'am': mutate_am, 'inter': mutate_inter}


class FDAModule(nn.Module):
    def __init__(self, mode='as', image_size=(512, 512)):
        super(FDAModule, self).__init__()
        if mode not in func_dict:
            raise NotImplementedError('FDA mutation method not implemented')
        self.mutate_func = func_dict[mode]
        self.image_size = image_size
        center_map = np.ones(image_size)
        center_map[list(range(1, image_size[0], 2))] *= -1
        center_map[np.ix_(list(range(image_size[0])), list(range(1, image_size[1], 2)))] *= -1
        self.center_map = torch.tensor(center_map, requires_grad=False)

    def forward(self, src_img, trg_img, l=0.1):  # todo 中心化这里还要确认一下
        # exchange magnitude
        # input: src_img, trg_img
        assert (src_img.shape[-2] == self.image_size[-2] and src_img.shape[-1] == self.image_size[-1])

        src_img_ = src_img.clone().detach()
        trg_img_ = trg_img.clone().detach()

        if self.center_map.device != src_img_.device:
            self.center_map = self.center_map.to(device=src_img_.device)
        src_img_ *= self.center_map
        trg_img_ *= self.center_map

        # get fft of both source and target
        fft_src = torch.fft.fft2(src_img_)
        fft_trg = torch.fft.fft2(trg_img_)

        # extract amplitude and phase of both ffts
        amp_src, pha_src = torch.abs(fft_src), torch.angle(fft_src)
        amp_trg, pha_trg = torch.abs(fft_trg), torch.angle(fft_trg)

        amp_src_ = self.mutate_func(amp_src.clone(), amp_trg.clone(), L=l)

        # recompose fft of source
        fft_src_ = amp_src_ * torch.exp(1j * pha_src)
        src_in_trg = torch.abs(torch.fft.ifft2(fft_src_)).type(torch.float32)

        return src_in_trg
