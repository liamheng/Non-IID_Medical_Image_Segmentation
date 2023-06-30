# -*- coding: UTF-8 -*-
"""
@Function:from two-stage to one-stage
@File: DG_one_model.py
@Date: 2021/9/14 20:45 
@Author: Hever
"""
import torch
import itertools

from utils import metrics
from utils.gaussian_mixup import GaussianMixUp
from .base_model import BaseModel
from . import networks
from models.guided_filter_pytorch.HFC_filter import HFCFilter


def hfc_mul_mask(hfc_filter, image, mask):
    hfc = hfc_filter(image, mask)
    # return hfc
    return (hfc + 1) * mask - 1
    # return image


class FreesdgSegmentationModel(BaseModel):
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        parser.set_defaults(norm='instance', netG='unet_combine_2layer', dataset_mode='aligned', no_dropout=True,
                            lr=0.001)
        if is_train:
            parser.add_argument('--lambda_high', type=float, default=1.0)
            parser.add_argument('--lambda_seg', type=float, default=1.0)
            parser.add_argument('--segmentation_loss', type=str, default='BCELoss')
        else:
            parser.add_argument('--metrics', type=str, default='f1,acc', )
            parser.add_argument('--confusion_threshold', type=float, default=0.5)
        parser.add_argument('--filter_width', type=int, default=27, help='weight for G loss')
        parser.add_argument('--nsig', type=int, default=9, help='weight for G loss')

        parser.add_argument('--attention_type', type=str, default='CBAMBlock')

        parser.add_argument('--no_hfc', action='store_true',
                            help='do not input hfc image into model, but original image (output image remains hfc)')

        parser.add_argument('--do_mixup', action='store_true')
        parser.add_argument('--mixup_size', type=int, default=100)

        parser.add_argument('--no_fact', action='store_true')

        return parser

    def __init__(self, opt):
        BaseModel.__init__(self, opt)

        self.no_hfc = opt.no_hfc  # whether to do hfc on input image
        self.hfc_filter = HFCFilter(opt.filter_width, opt.nsig, sub_low_ratio=1, sub_mask=True, is_clamp=True).to(
            self.device)
        self.do_mixup = opt.do_mixup
        if self.do_mixup:
            self.mixup_filter = GaussianMixUp(mixup_size=opt.mixup_size, sub_low_ratio=1, sub_mask=True,
                                              is_clamp=True).to(self.device)
        self.no_fact = opt.no_fact

        self.loss_names = ['G_high', 'G_seg', 'G']

        self.visual_names_train = ['image_original', 'high_original', 'high_out', 'seg_out', 'seg_label',
                                   'mask'] + [] if self.no_fact else ['image_fact', 'high_fact', 'image_target']
        self.visual_names_test = ['image_original', 'high_original', 'high_out', 'seg_out', 'seg_label', 'seg_out_binary']
        if self.isTrain:
            self.model_names = ['G']
            self.visual_names = self.visual_names_train
        else:
            self.model_names = ['G']
            self.visual_names = self.visual_names_test

        self.netG = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.norm,
                                      not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids,
                                      last_layer='Sigmoid', attention_type=opt.attention_type)

        if self.isTrain:
            self.criterion_segmentation = getattr(torch.nn, opt.segmentation_loss)()
            self.criterionL1 = torch.nn.L1Loss()
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_G)
            self.lambda_high = opt.lambda_high
            self.lambda_seg = opt.lambda_seg
        else:
            self.confusion_matrix = metrics.Metric(opt.output_nc, threshold=opt.confusion_threshold)

    def set_input(self, input, isTrain=None):
        if not self.isTrain or isTrain is not None:
            self.image_original = input['image_original'].to(self.device)
            self.mask = input['mask'].to(self.device)
            self.image_paths = input['source_path']
            self.seg_label = input['label'].to(self.device)
            self.high_original = hfc_mul_mask(self.hfc_filter, self.image_original, self.mask)
            if self.opt.ignore_od:
                self.mask_od = input['mask_od'].to(device=self.device)
        else:
            self.image_original = input['image_original'].to(self.device)
            self.mask = input['mask'].to(self.device)
            self.seg_label = input['label'].to(self.device)
            self.image_paths = input['source_path']
            if self.no_fact:
                self.high_original = hfc_mul_mask(self.hfc_filter, self.image_original, self.mask)
                self.high_input = hfc_mul_mask(self.mixup_filter if self.do_mixup else self.hfc_filter, self.image_original,
                                          self.mask)
            else:
                self.image_target = input['target']
                self.image_fact = input['image_fact'].to(self.device)
                self.high_original = hfc_mul_mask(self.hfc_filter, self.image_original, self.mask)
                self.high_fact = hfc_mul_mask(self.mixup_filter if self.do_mixup else self.hfc_filter, self.image_fact,
                                          self.mask)

    def forward(self):
        if self.no_fact:
            self.high_out, self.seg_out = self.netG(self.image_original if self.no_hfc else self.high_input)
        else:
            self.high_out, self.seg_out = self.netG(self.image_fact if self.no_hfc else self.high_fact)
        self.high_out = (self.high_out + 1) * self.mask - 1
        self.seg_out = self.seg_out * self.mask

    def compute_visuals(self):
        self.seg_label = self.seg_label * 2 - 1
        self.seg_out = self.seg_out * 2 - 1
        self.mask = self.mask * 2 - 1
        if self.isTrain:
            self.image_target = self.image_target * 2 - 1
        if not self.isTrain:
            self.seg_out_binary = self.seg_out_binary * 2 - 1

    def test(self):
        self.visual_names = self.visual_names_test
        with torch.no_grad():
            # For visualisation
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
        self.visual_names = self.visual_names_train
        for name in self.model_names:
            if isinstance(name, str):
                net = getattr(self, 'net' + name)
                net.train()

    def backward_G(self):
        self.loss_G_high = self.criterionL1(self.high_original, self.high_out) * self.lambda_high
        self.loss_G_seg = self.criterion_segmentation(self.seg_out, self.seg_label) * self.lambda_seg
        self.loss_G = self.loss_G_high + self.loss_G_seg
        self.loss_G.backward()

    def optimize_parameters(self):
        self.forward()
        self.optimizer_G.zero_grad()
        self.backward_G()
        self.optimizer_G.step()

    def get_metric_results(self):
        results = self.confusion_matrix.evaluate()
        metrics_list = self.opt.metrics.split(',')
        return {name: results[name][1].item() for name in metrics_list}
