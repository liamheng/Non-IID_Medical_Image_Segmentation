import torch
import torch.nn as nn
import functools

from models.attention_modules.CBAM import CBAMBlock
import models.attention_modules


class UnetCombine2LayerAttentionGenerator(nn.Module):
    """Create a Unet-based generator"""

    def __init__(self, input_nc, output_nc, ngf_list=(64, 128, 256, 512, 512, 512, 512, 512),
                 norm_layer=nn.BatchNorm2d, use_dropout=False, last_layer='Tanh', attention_type='CBAMBlock'):

        super(UnetCombine2LayerAttentionGenerator, self).__init__()

        self.conv_num = len(ngf_list) - 1
        self.unet_block_list = nn.ModuleList()
        self.attention_list = nn.ModuleList()
        for i in range(self.conv_num):
            inner_most = i == self.conv_num - 1
            outer_most = i == 0
            self.unet_block_list.append(UnetSkipConnectionBlock(inner_nc=ngf_list[i],
                                                                input_nc=input_nc if outer_most else None,
                                                                outer_nc=output_nc if outer_most else ngf_list[i - 1],
                                                                norm_layer=norm_layer, last_layer=last_layer,
                                                                outermost=outer_most, innermost=inner_most,
                                                                use_dropout=False if inner_most or outer_most else use_dropout))
            if i != self.conv_num - 1:
                self.attention_list.append(models.attention_modules.find_model_using_name(attention_type)(2 * ngf_list[i]))

    def forward(self, x):
        # downsample
        down_out_list = []
        for i in range(self.conv_num):
            x = self.unet_block_list[i].down(x)
            down_out_list.append(x)

        # upsample 1
        up_out_list_1 = []
        for i in range(self.conv_num):
            x = self.unet_block_list[-1 - i].h_up(
                down_out_list[-1] if i == 0 else torch.cat([x, down_out_list[-1 - i]], 1))
            up_out_list_1.append(x)

        # upsample 2
        up_out_list_2 = []
        for i in range(self.conv_num):
            input_val = down_out_list[-1] if i == 0 else torch.cat([up_out_list_1[i - 1], x], 1)
            x = self.unet_block_list[-1 - i].up(input_val if i == 0 else self.attention_list[-i](input_val))
            up_out_list_2.append(x)

        return up_out_list_1[-1], up_out_list_2[-1]


class UnetSkipConnectionBlock(nn.Module):

    def __init__(self, outer_nc, inner_nc, input_nc=None,
                 outermost=False, innermost=False, norm_layer=nn.BatchNorm2d, use_dropout=False, last_layer='Tanh'):
        super(UnetSkipConnectionBlock, self).__init__()
        self.outermost = outermost
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        if input_nc is None:
            input_nc = outer_nc
        downconv = nn.Conv2d(input_nc, inner_nc, kernel_size=4,
                             stride=2, padding=1, bias=use_bias)
        downrelu = nn.LeakyReLU(0.2)
        downnorm = norm_layer(inner_nc)
        uprelu = nn.ReLU()
        upnorm = norm_layer(outer_nc)
        h_uprelu = nn.ReLU()
        h_upnorm = norm_layer(outer_nc)

        if outermost:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1)
            # 仅仅修改了这个
            h_upconv = nn.ConvTranspose2d(inner_nc * 2, 3,
                                          kernel_size=4, stride=2,
                                          padding=1)
            down = [downconv]
            up = [uprelu, upconv, getattr(torch.nn, last_layer)()]
            h_up = [h_uprelu, h_upconv, nn.Tanh()]
        elif innermost:
            upconv = nn.ConvTranspose2d(inner_nc, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias)
            h_upconv = nn.ConvTranspose2d(inner_nc, outer_nc,
                                          kernel_size=4, stride=2,
                                          padding=1, bias=use_bias)
            down = [downrelu, downconv]
            up = [uprelu, upconv, upnorm]
            h_up = [h_uprelu, h_upconv, h_upnorm]
        else:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias)
            h_upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                          kernel_size=4, stride=2,
                                          padding=1, bias=use_bias)
            down = [downrelu, downconv, downnorm]
            up = [uprelu, upconv, upnorm]
            h_up = [h_uprelu, h_upconv, h_upnorm]
            if use_dropout:
                up = up + [nn.Dropout(0.5)]
                h_up = h_up + [nn.Dropout(0.5)]
        self.up = nn.Sequential(*up)
        self.h_up = nn.Sequential(*h_up)
        self.down = nn.Sequential(*down)
