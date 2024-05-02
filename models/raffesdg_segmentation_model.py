# -*- coding: UTF-8 -*-
"""
@Function:from two-stage to one-stage
@File: DG_one_model.py
@Date: 2021/9/14 20:45 
@Author: Hever
"""

import torch

from torchvision.transforms import transforms

from utils.augmentations import FMAug
from .base_model import BaseModel
from . import networks
from models.guided_filter_pytorch.HFC_filter import HFCFilter
from .guided_filter_pytorch.ModifiedHFCFilter import FourierButterworthHFCFilter


def hfc_mul_mask(hfc_filter, image, mask, do_norm=False):
    # print('image', image.min(), image.max())
    hfc = hfc_filter((image / 2 + 0.5), mask)
    if do_norm:
        hfc = 2 * hfc - 1
    # return hfc
    return (hfc + 1) * mask - 1
    # return image


class RAFFESDGSEGMENTATIONModel(BaseModel):
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        parser.set_defaults(norm='instance', netG='unet_combine_2layer', dataset_mode='aligned',
                            lr=0.001)
        if is_train:
            parser.add_argument('--lambda_high', type=float, default=1.0)
            parser.add_argument('--lambda_seg', type=float, default=1.0)
            parser.add_argument('--segmentation_loss', type=str, default='BCELoss')
            parser.add_argument('--reconstruction_loss', type=str, default='L1Loss')
        else:
            parser.add_argument('--metrics', type=str, default='f1,acc', )
            parser.add_argument('--confusion_threshold', type=float, default=0.5)

        parser.add_argument('--reverse_last', action='store_true')
        parser.add_argument('--double_conv', action='store_true')
        parser.add_argument('--dropout_type', type=int, default=1)
        parser.add_argument('--first_attention', action='store_true')
        parser.add_argument('--second_attention', action='store_true')

        parser.add_argument('--filter_width', type=int, default=27, help='weight for G loss')
        parser.add_argument('--nsig', type=int, default=9, help='weight for G loss')

        parser.add_argument('--attention_type', type=str, default='CBAMBlock')

        parser.add_argument('--no_hfc', action='store_true',
                            help='do not input hfc image into model, but original image (output image remains hfc)')
        parser.add_argument('--no_hfc_output', action='store_true')

        parser.add_argument('--do_mixup', action='store_true')
        parser.add_argument('--mixup_type', type=str, nargs='+', default=['patch', 'grid', 'smooth'])
        parser.add_argument('--mixup_hfc_type', type=str, default='butterworth_per_channel')
        parser.add_argument('--mixup_remain_original', action='store_true')

        return parser

    def __init__(self, opt):
        BaseModel.__init__(self, opt)

        self.no_hfc = opt.no_hfc
        self.no_hfc_output = opt.no_hfc_output
        self.do_mixup = opt.do_mixup

        if self.do_mixup:
            if self.isTrain:
                mixup_type_list = [opt.mixup_type] if isinstance(opt.mixup_type, str) else opt.mixup_type
                self.fmaug = FMAug(mixup_type=','.join(mixup_type_list), hfc_type=opt.mixup_hfc_type,
                                   remain_original=str(opt.mixup_remain_original), device=str(self.device))
                self.norm_func = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                self.hfc_filter_out = HFCFilter(opt.filter_width, opt.nsig, sub_low_ratio=1, sub_mask=True,
                                                is_clamp=True).to(self.device)
            else:
                self.hfc_filter_in = FourierButterworthHFCFilter(butterworth_d0_ratio=0.005, butterworth_n=1,
                                                                 do_median_padding=False, image_size=(512, 512)).to(
                    device=self.device)

        # when do_mixup is true, either no_hfc nor no_hfc_output could be true
        if not self.no_hfc:
            self.hfc_filter_in = FourierButterworthHFCFilter(butterworth_d0_ratio=0.005, butterworth_n=1,
                                                             do_median_padding=False, image_size=(512, 512)).to(
                device=self.device)
        if not self.no_hfc_output:
            self.hfc_filter_out = HFCFilter(opt.filter_width, opt.nsig, sub_low_ratio=1, sub_mask=True,
                                            is_clamp=True).to(self.device)

        self.loss_names = ['G', 'G_seg'] + ([] if self.no_hfc_output else ['G_high'])

        self.model_names = ['G']
        if self.isTrain:
            self.visual_names = ['image_original', 'high_original_in', 'seg_out', 'seg_label', 'mask'] + (
                [] if self.no_hfc_output else ['high_original', 'high_out'])
        else:
            self.visual_names = ['image_original', 'seg_out', 'seg_label', 'seg_out_binary', 'out_seg', 'out_seg_binary'] + (
                [] if self.no_hfc_output else ['high_original', 'high_out'])

        self.netG = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.norm,
                                      (not opt.no_dropout) * opt.dropout_type, opt.init_type, opt.init_gain,
                                      self.gpu_ids,
                                      last_layer='Sigmoid', attention_type=opt.attention_type,
                                      reverse_last=opt.reverse_last, first_attention=opt.first_attention,
                                      second_attention=opt.second_attention)

        if self.isTrain:
            # define loss functions
            self.criterion_segmentation = getattr(torch.nn, opt.segmentation_loss)()
            self.criterion_rec = getattr(torch.nn, opt.reconstruction_loss)()
            # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_G)
            self.lambda_high = opt.lambda_high
            self.lambda_seg = opt.lambda_seg

    def mixup_filter(self, image, mask):
        result = self.fmaug(image, mask).transpose(0, 3, 1, 2)
        return self.norm_func(torch.FloatTensor(result)).to(device=self.device)

    def set_input(self, input, isTrain=None):
        if not self.isTrain or isTrain is not None:
            self.image_original = input['image_original'].to(self.device)
            self.mask = input['mask'].to(self.device)
            self.image_paths = input['source_path']
            self.seg_label = input['label'].to(self.device)
            self.high_original = hfc_mul_mask(self.hfc_filter_in, self.image_original, self.mask, do_norm=True)
            if self.opt.ignore_od:
                self.mask_od = input['mask_od'].to(device=self.device)
        else:
            self.image_original = input['image_original'].to(self.device)
            self.mask = input['mask'].to(self.device)
            self.seg_label = input['label'].to(self.device)
            self.image_paths = input['source_path']
            self.high_original = hfc_mul_mask(self.hfc_filter_out, self.image_original, self.mask)
            self.high_original_in = hfc_mul_mask(self.mixup_filter if self.do_mixup else self.hfc_filter_in,
                                                 self.image_original, self.mask)

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        if self.no_hfc_output:
            self.seg_out = self.netG(self.image_original if self.no_hfc else self.high_original_in)
        else:
            if self.opt.reverse_last:
                self.seg_out, self.high_out = self.netG(self.image_original if self.no_hfc else self.high_original_in)
            else:
                self.high_out, self.seg_out = self.netG(self.image_original if self.no_hfc else self.high_original_in)
            self.high_out = (self.high_out + 1) * self.mask - 1
        # print(self.seg_out.shape)
        self.seg_out = self.seg_out * self.mask

    def compute_visuals(self):
        self.seg_label = self.seg_label * 2 - 1
        self.seg_out = self.seg_out * 2 - 1
        self.mask = self.mask * 2 - 1
        self.out_seg = self.seg_out
        if not self.isTrain:
            self.seg_out_binary = self.seg_out_binary * 2 - 1
            self.out_seg_binary = self.seg_out_binary

    def test(self):
        with torch.no_grad():
            # For visualisation
            if self.no_hfc_output:
                self.seg_out = self.netG(self.image_original if self.no_hfc else self.high_original)
            else:
                if self.opt.reverse_last:
                    self.seg_out, self.high_out = self.netG(self.image_original if self.no_hfc else self.high_original)
                else:
                    self.high_out, self.seg_out = self.netG(self.image_original if self.no_hfc else self.high_original)
                self.high_out = (self.high_out + 1) * self.mask - 1
            self.seg_out = self.seg_out * self.mask

            if self.opt.ignore_od:
                self.seg_out = self.seg_out * self.mask_od
                self.seg_label = self.seg_label * self.mask_od

            self.confusion_matrix.update(self.seg_out, self.seg_label)

            self.seg_out_binary = self.seg_out > 0.5

            self.compute_visuals()

    def train(self):
        """Make models eval mode during test time"""
        for name in self.model_names:
            if isinstance(name, str):
                net = getattr(self, 'net' + name)
                net.train()

    def backward_G(self):
        self.loss_G_seg = self.criterion_segmentation(self.seg_out, self.seg_label) * self.lambda_seg

        if not self.no_hfc_output:
            self.loss_G_high = self.criterion_rec(self.high_original, self.high_out) * self.lambda_high
            self.loss_G = self.loss_G_high + self.loss_G_seg
        else:
            self.loss_G = self.loss_G_seg

        self.loss_G.backward()

    def optimize_parameters(self):
        self.forward()
        self.optimizer_G.zero_grad()
        self.backward_G()
        self.optimizer_G.step()
