import torch
import torch.nn as nn
import functools
import torch.nn.functional as F
from torch.nn import init
from torch.optim import lr_scheduler
from .guided_filter_pytorch.guided_filter import FastGuidedFilter
from models.backbone.unet_combine_2layer import UnetCombine2LayerGenerator
from models.backbone.unet_combine_2layer_attention import UnetCombine2LayerAttentionGenerator


###############################################################################
# Helper Functions
###############################################################################


class Identity(nn.Module):
    def forward(self, x):
        return x


def get_norm_layer(norm_type='instance'):
    """Return a normalization layer

    Parameters:
        norm_type (str) -- the name of the normalization layer: batch | instance | none

    For BatchNorm, we use learnable affine parameters and track running statistics (mean/stddev).
    For InstanceNorm, we do not use learnable affine parameters. We do not track running statistics.
    """
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True, track_running_stats=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=False)
    elif norm_type == 'none':
        def norm_layer(x):
            return Identity()
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer


def get_scheduler(optimizer, opt):
    """Return a learning rate scheduler
    制作一个学习规划器，控制学习率
    Parameters:
        optimizer          -- the optimizer of the network
        opt (option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions．　
                              opt.lr_policy is the name of learning rate policy: linear | step | plateau | cosine

    For 'linear', we keep the same learning rate for the first <opt.n_epochs> epochs
    and linearly decay the rate to zero over the next <opt.n_epochs_decay> epochs.
    For other schedulers (step, plateau, and cosine), we use the default PyTorch schedulers.
    See https://pytorch.org/docs/stable/optim.html for more details.
    """
    if opt.lr_policy == 'linear':
        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch + opt.epoch_count - opt.n_epochs) / float(opt.n_epochs_decay + 1)
            return lr_l

        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif opt.lr_policy == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.lr_decay_iters, gamma=0.1)
    elif opt.lr_policy == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
    elif opt.lr_policy == 'cosine':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=opt.n_epochs, eta_min=0)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', opt.lr_policy)
    return scheduler


def init_weights(net, init_type='normal', init_gain=0.02, verbose=False):
    """Initialize network weights.

    Parameters:
        net (network)   -- network to be initialized
        init_type (str) -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        init_gain (float)    -- scaling factor for normal, xavier and orthogonal.

    We use 'normal' in the original pix2pix and CycleGAN paper. But xavier and kaiming might
    work better for some applications. Feel free to try yourself.
    """

    def init_func(m):  # define the initialization function
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find(
                'BatchNorm2d') != -1:  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)

    if not verbose:
        print('initialize network with %s' % init_type)
    net.apply(init_func)  # apply the initialization function <init_func>


def init_net(net, init_type='normal', init_gain=0.02, gpu_ids=[], verbose=False):
    """Initialize a network: 1. register CPU/GPU device (with multi-GPU support); 2. initialize the network weights
    Parameters:
        net (network)      -- the network to be initialized
        init_type (str)    -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        gain (float)       -- scaling factor for normal, xavier and orthogonal.
        gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2

    Return an initialized network.
    """
    if len(gpu_ids) > 0:
        assert (torch.cuda.is_available())
        net.to(gpu_ids[0])
        net = torch.nn.DataParallel(net, gpu_ids)  # multi-GPUs
    init_weights(net, init_type, init_gain=init_gain, verbose=verbose)
    return net


def define_G(input_nc, output_nc, ngf, netG, norm='batch', use_dropout=False, init_type='normal', init_gain=0.02,
             gpu_ids=[], last_layer='Tanh', verbose=False, original_dense=False, attention_type='CBAMBlock'):
    """Create a generator

    Parameters:
        input_nc (int) -- the number of channels in input images
        output_nc (int) -- the number of channels in output images
        ngf (int) -- the number of filters in the last conv layer
        netG (str) -- the architecture's name: resnet_9blocks | resnet_6blocks | unet_256 | unet_128
        norm (str) -- the name of normalization layers used in the network: batch | instance | none
        use_dropout (bool) -- if use dropout layers.
        init_type (str)    -- the name of our initialization method.
        init_gain (float)  -- scaling factor for normal, xavier and orthogonal.
        gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2

    Returns a generator

    Our current implementation provides two types of generators:
        U-Net: [unet_128] (for 128x128 input images) and [unet_256] (for 256x256 input images)
        The original U-Net paper: https://arxiv.org/abs/1505.04597

        Resnet-based generator: [resnet_6blocks] (with 6 Resnet blocks) and [resnet_9blocks] (with 9 Resnet blocks)
        Resnet-based generator consists of several Resnet blocks between a few downsampling/upsampling operations.
        We adapt Torch code from Justin Johnson's neural style transfer project (https://github.com/jcjohnson/fast-neural-style).


    The generator has been initialized by <init_net>. It uses RELU for non-linearity.
    """
    net = None
    norm_layer = get_norm_layer(norm_type=norm)

    if netG == 'resnet_9blocks':
        net = ResnetGenerator(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=9,
                              last_layer=last_layer)
    elif netG == 'resnet_6blocks':
        net = ResnetGenerator(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=6,
                              last_layer=last_layer)
    elif netG == 'unet_128':
        net = UnetGenerator(input_nc, output_nc, 7, ngf, norm_layer=norm_layer, use_dropout=use_dropout,
                            last_layer=last_layer)
    elif netG == 'unet_256':
        net = UnetGenerator(input_nc, output_nc, 8, ngf, norm_layer=norm_layer, use_dropout=use_dropout,
                            last_layer=last_layer)
    elif netG == 'unet_256_vector':
        net = UnetVectorGenerator(input_nc, output_nc, 8, ngf, norm_layer=norm_layer, use_dropout=use_dropout,
                                  last_layer=last_layer)
    elif netG == 'unet_combine_2layer':
        net = UnetCombine2LayerGenerator(input_nc, output_nc, 8, ngf, norm_layer=norm_layer, use_dropout=use_dropout,
                                         last_layer=last_layer)
    elif netG == 'unet_mo':
        net = UnetMO(input_nc, output_nc, ngf=ngf, norm_layer=norm_layer, use_dropout=use_dropout,
                     last_layer=last_layer)
    elif netG == 'unet_cascade':
        net = UnetCascade(input_nc, output_nc, ngf=ngf, norm_layer=norm_layer, use_dropout=use_dropout,
                          last_layer=last_layer, original_dense=original_dense)
    elif netG == 'unet_combine_2layer_attention':
        net = UnetCombine2LayerAttentionGenerator(input_nc, output_nc, norm_layer=norm_layer, use_dropout=use_dropout,
                                                  last_layer=last_layer, attention_type=attention_type)
    elif netG == 'naive_autoencoder':
        net = NaiveAutoEncoder()
    else:
        raise NotImplementedError('Generator model name [%s] is not recognized' % netG)
    return init_net(net, init_type, init_gain, gpu_ids, verbose)


def define_D(input_nc, ndf, netD, n_layers_D=3, norm='batch', init_type='normal', init_gain=0.02, gpu_ids=[]):
    """Create a discriminator

    Parameters:
        input_nc (int)     -- the number of channels in input images
        ndf (int)          -- the number of filters in the first conv layer
        netD (str)         -- the architecture's name: basic | n_layers | pixel
        n_layers_D (int)   -- the number of conv layers in the discriminator; effective when netD=='n_layers'
        norm (str)         -- the type of normalization layers used in the network.
        init_type (str)    -- the name of the initialization method.
        init_gain (float)  -- scaling factor for normal, xavier and orthogonal.
        gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2

    Returns a discriminator

    Our current implementation provides three types of discriminators:
        [basic]: 'PatchGAN' classifier described in the original pix2pix paper.
        It can classify whether 70×70 overlapping patches are real or fake.
        Such a patch-level discriminator architecture has fewer parameters
        than a full-image discriminator and can work on arbitrarily-sized images
        in a fully convolutional fashion.

        [n_layers]: With this mode, you can specify the number of conv layers in the discriminator
        with the parameter <n_layers_D> (default=3 as used in [basic] (PatchGAN).)

        [pixel]: 1x1 PixelGAN discriminator can classify whether a pixel is real or not.
        It encourages greater color diversity but has no effect on spatial statistics.

    The discriminator has been initialized by <init_net>. It uses Leakly RELU for non-linearity.
    """
    net = None
    norm_layer = get_norm_layer(norm_type=norm)

    if netD == 'basic':  # default PatchGAN classifier
        net = NLayerDiscriminator(input_nc, ndf, n_layers=3, norm_layer=norm_layer)
    elif netD == 'n_layers':  # more options
        net = NLayerDiscriminator(input_nc, ndf, n_layers_D, norm_layer=norm_layer)
    elif netD == 'pixel':  # classify if each pixel is real or fake
        net = PixelDiscriminator(input_nc, ndf, norm_layer=norm_layer)
    elif netD == 'Conv':  # classify if each pixel is real or fake
        net = ConvDiscriminator(512, ndf, norm_layer=norm_layer)
    elif netD == 'FC':  # classify if each pixel is real or fake
        net = FCDiscriminator(input_nc, ndf)
    else:
        raise NotImplementedError('Discriminator model name [%s] is not recognized' % netD)
    return init_net(net, init_type, init_gain, gpu_ids)


##############################################################################
# Classes
##############################################################################
class GANLoss(nn.Module):
    """Define different GAN objectives.
    真就是真，假就是假，这时loss就正常计算
    但若一真一假
    The GANLoss class abstracts away the need to create the target label tensor
    that has the same size as the input.
    """

    def __init__(self, gan_mode, target_real_label=1.0, target_fake_label=0.0):
        """ Initialize the GANLoss class.

        Parameters:
            gan_mode (str) - - the type of GAN objective. It currently supports vanilla, lsgan, and wgangp.
            target_real_label (bool) - - label for a real image
            target_fake_label (bool) - - label of a fake image

        Note: Do not use Sigmoid as the last layer of Discriminator.
        LSGAN needs no Sigmoid. vanilla GANs will handle it with BCEWithLogitsLoss.
        """
        super(GANLoss, self).__init__()
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))
        self.gan_mode = gan_mode
        if gan_mode == 'lsgan':
            self.loss = nn.MSELoss()
        elif gan_mode == 'vanilla':
            self.loss = nn.BCEWithLogitsLoss()
        elif gan_mode in ['wgangp']:
            self.loss = None
        else:
            raise NotImplementedError('gan mode %s not implemented' % gan_mode)

    def get_target_tensor(self, prediction, target_is_real):
        """Create label tensors with the same size as the input.

        Parameters:
            prediction (tensor) - - tpyically the prediction from a discriminator
            target_is_real (bool) - - if the ground truth label is for real images or fake images

        Returns:
            A label tensor filled with ground truth label, and with the size of the input
        """

        if target_is_real:
            target_tensor = self.real_label
        else:
            target_tensor = self.fake_label
        return target_tensor.expand_as(prediction)

    def __call__(self, prediction, target_is_real):
        """Calculate loss given Discriminator's output and grount truth labels.

        Parameters:
            prediction (tensor) - - tpyically the prediction output from a discriminator
            target_is_real (bool) - - if the ground truth label is for real images or fake images

        Returns:
            the calculated loss.
        """
        if self.gan_mode in ['lsgan', 'vanilla']:
            target_tensor = self.get_target_tensor(prediction, target_is_real)
            # 真和真，假和假，算出的loss可以直接minimize
            loss = self.loss(prediction, target_tensor)
        elif self.gan_mode == 'wgangp':
            if target_is_real:
                loss = -prediction.mean()
            else:
                loss = prediction.mean()
        return loss


def cal_gradient_penalty(netD, real_data, fake_data, device, type='mixed', constant=1.0, lambda_gp=10.0):
    """Calculate the gradient penalty loss, used in WGAN-GP paper https://arxiv.org/abs/1704.00028

    Arguments:
        netD (network)              -- discriminator network
        real_data (tensor array)    -- real images
        fake_data (tensor array)    -- generated images from the generator
        device (str)                -- GPU / CPU: from torch.device('cuda:{}'.format(self.gpu_ids[0])) if self.gpu_ids else torch.device('cpu')
        type (str)                  -- if we mix real and fake data or not [real | fake | mixed].
        constant (float)            -- the constant used in formula ( ||gradient||_2 - constant)^2
        lambda_gp (float)           -- weight for this loss

    Returns the gradient penalty loss
    """
    if lambda_gp > 0.0:
        if type == 'real':  # either use real images, fake images, or a linear interpolation of two.
            interpolatesv = real_data
        elif type == 'fake':
            interpolatesv = fake_data
        elif type == 'mixed':
            alpha = torch.rand(real_data.shape[0], 1, device=device)
            alpha = alpha.expand(real_data.shape[0], real_data.nelement() // real_data.shape[0]).contiguous().view(
                *real_data.shape)
            interpolatesv = alpha * real_data + ((1 - alpha) * fake_data)
        else:
            raise NotImplementedError('{} not implemented'.format(type))
        interpolatesv.requires_grad_(True)
        disc_interpolates = netD(interpolatesv)
        gradients = torch.autograd.grad(outputs=disc_interpolates, inputs=interpolatesv,
                                        grad_outputs=torch.ones(disc_interpolates.size()).to(device),
                                        create_graph=True, retain_graph=True, only_inputs=True)
        gradients = gradients[0].view(real_data.size(0), -1)  # flat the data
        gradient_penalty = (((gradients + 1e-16).norm(2, dim=1) - constant) ** 2).mean() * lambda_gp  # added eps
        return gradient_penalty, gradients
    else:
        return 0.0, None


class ResnetGenerator(nn.Module):
    """Resnet-based generator that consists of Resnet blocks between a few downsampling/upsampling operations.

    We adapt Torch code and idea from Justin Johnson's neural style transfer project(https://github.com/jcjohnson/fast-neural-style)
    """

    def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, n_blocks=6,
                 padding_type='reflect', last_layer='Tanh'):
        """Construct a Resnet-based generator

        Parameters:
            input_nc (int)      -- the number of channels in input images
            output_nc (int)     -- the number of channels in output images
            ngf (int)           -- the number of filters in the last conv layer
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers
            n_blocks (int)      -- the number of ResNet blocks
            padding_type (str)  -- the name of padding layer in conv layers: reflect | replicate | zero
        """
        assert (n_blocks >= 0)
        super(ResnetGenerator, self).__init__()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        model = [nn.ReflectionPad2d(3),
                 nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0, bias=use_bias),
                 norm_layer(ngf),
                 nn.ReLU(True)]

        n_downsampling = 2
        for i in range(n_downsampling):  # add downsampling layers
            mult = 2 ** i
            model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1, bias=use_bias),
                      norm_layer(ngf * mult * 2),
                      nn.ReLU(True)]

        mult = 2 ** n_downsampling
        for i in range(n_blocks):  # add ResNet blocks

            model += [ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout,
                                  use_bias=use_bias)]

        for i in range(n_downsampling):  # add upsampling layers
            mult = 2 ** (n_downsampling - i)
            model += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2),
                                         kernel_size=3, stride=2,
                                         padding=1, output_padding=1,
                                         bias=use_bias),
                      norm_layer(int(ngf * mult / 2)),
                      nn.ReLU(True)]
        model += [nn.ReflectionPad2d(3)]
        model += [nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0)]
        if last_layer == 'Tanh':
            model += [nn.Tanh()]
        elif last_layer == 'Sigmoid':
            model += [nn.Sigmoid()]

        self.model = nn.Sequential(*model)

    def forward(self, input):
        """Standard forward"""
        return self.model(input)


class ResnetBlock(nn.Module):
    """Define a Resnet block"""

    def __init__(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        """Initialize the Resnet block

        A resnet block is a conv block with skip connections
        We construct a conv block with build_conv_block function,
        and implement skip connections in <forward> function.
        Original Resnet paper: https://arxiv.org/pdf/1512.03385.pdf
        """
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, use_dropout, use_bias)

    def build_conv_block(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        """Construct a convolutional block.

        Parameters:
            dim (int)           -- the number of channels in the conv layer.
            padding_type (str)  -- the name of padding layer: reflect | replicate | zero
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers.
            use_bias (bool)     -- if the conv layer uses bias or not

        Returns a conv block (with a conv layer, a normalization layer, and a non-linearity layer (ReLU))
        """
        conv_block = []
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias), norm_layer(dim), nn.ReLU(True)]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias), norm_layer(dim)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        """Forward function (with skip connections)"""
        out = x + self.conv_block(x)  # add skip connections
        return out


class UnetGenerator(nn.Module):
    """Create a Unet-based generator"""

    def __init__(self, input_nc, output_nc, num_downs, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False,
                 last_layer='Tanh'):
        """Construct a Unet generator
        Parameters:
            input_nc (int)  -- the number of channels in input images
            output_nc (int) -- the number of channels in output images
            num_downs (int) -- the number of downsamplings in UNet. For example, # if |num_downs| == 7,
                                image of size 128x128 will become of size 1x1 # at the bottleneck
            ngf (int)       -- the number of filters in the last conv layer
            norm_layer      -- normalization layer

        We construct the U-Net from the innermost layer to the outermost layer.
        It is a recursive process.
        """
        super(UnetGenerator, self).__init__()
        # construct unet structure
        unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=None, norm_layer=norm_layer,
                                             innermost=True)  # add the innermost layer
        for i in range(num_downs - 5):  # add intermediate layers with ngf * 8 filters
            unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=unet_block,
                                                 norm_layer=norm_layer, use_dropout=use_dropout)
        # gradually reduce the number of filters from ngf * 8 to ngf
        unet_block = UnetSkipConnectionBlock(ngf * 4, ngf * 8, input_nc=None, submodule=unet_block,
                                             norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(ngf * 2, ngf * 4, input_nc=None, submodule=unet_block,
                                             norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(ngf, ngf * 2, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        self.model = UnetSkipConnectionBlock(output_nc, ngf, input_nc=input_nc, submodule=unet_block, outermost=True,
                                             norm_layer=norm_layer, last_layer=last_layer)  # add the outermost layer

    def forward(self, input):
        """Standard forward"""
        return self.model(input)


class UnetSkipConnectionBlock(nn.Module):
    """Defines the Unet submodule with skip connection.
        X -------------------identity----------------------
        |-- downsampling -- |submodule| -- upsampling --|
    """

    def __init__(self, outer_nc, inner_nc, input_nc=None,
                 submodule=None, outermost=False, innermost=False, norm_layer=nn.BatchNorm2d, use_dropout=False,
                 last_layer='Tanh'):
        """Construct a Unet submodule with skip connections.

        Parameters:
            outer_nc (int) -- the number of filters in the outer conv layer
            inner_nc (int) -- the number of filters in the inner conv layer
            input_nc (int) -- the number of channels in input images/features
            submodule (UnetSkipConnectionBlock) -- previously defined submodules
            outermost (bool)    -- if this module is the outermost module
            innermost (bool)    -- if this module is the innermost module
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers.
        """
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
        downrelu = nn.LeakyReLU(0.2, True)
        downnorm = norm_layer(inner_nc)
        uprelu = nn.ReLU(True)
        upnorm = norm_layer(outer_nc)

        if outermost:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1)
            down = [downconv]
            last = None
            if last_layer == 'Sigmoid':
                last = nn.Sigmoid()
            elif last_layer == 'Tanh':
                last = nn.Tanh()
            up = [uprelu, upconv, last]
            model = down + [submodule] + up
        elif innermost:
            upconv = nn.ConvTranspose2d(inner_nc, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias)
            down = [downrelu, downconv]
            up = [uprelu, upconv, upnorm]
            model = down + up
        else:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias)
            down = [downrelu, downconv, downnorm]
            up = [uprelu, upconv, upnorm]

            if use_dropout:
                model = down + [submodule] + up + [nn.Dropout(0.5)]
            else:
                model = down + [submodule] + up

        self.model = nn.Sequential(*model)

    def forward(self, x):
        if self.outermost:
            return self.model(x)
        else:  # add skip connections
            return torch.cat([x, self.model(x)], 1)


class UnetVectorGenerator(nn.Module):
    """Create a Unet-based generator"""

    def __init__(self, input_nc, output_nc, num_downs, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False,
                 inner_conv=True, last_layer='Tanh'):
        """Construct a Unet generator
        Parameters:
            input_nc (int)  -- the number of channels in input images
            output_nc (int) -- the number of channels in output images
            num_downs (int) -- the number of downsamplings in UNet. For example, # if |num_downs| == 7,
                                image of size 128x128 will become of size 1x1 # at the bottleneck
            ngf (int)       -- the number of filters in the last conv layer
            norm_layer      -- normalization layer

        We construct the U-Net from the innermost layer to the outermost layer.
        It is a recursive process.
        """
        super(UnetVectorGenerator, self).__init__()
        # construct unet structure
        unet_block = UnetSkipConnectionVectorBlock(ngf * 8, ngf * 8, input_nc=None, submodule=None,
                                                   norm_layer=norm_layer, innermost=True,
                                                   inner_conv=inner_conv)  # add the innermost layer
        self.innermost_block = unet_block
        for i in range(num_downs - 5):  # add intermediate layers with ngf * 8 filters
            unet_block = UnetSkipConnectionVectorBlock(ngf * 8, ngf * 8, input_nc=None, submodule=unet_block,
                                                       norm_layer=norm_layer, use_dropout=use_dropout)
        # gradually reduce the number of filters from ngf * 8 to ngf
        unet_block = UnetSkipConnectionVectorBlock(ngf * 4, ngf * 8, input_nc=None, submodule=unet_block,
                                                   norm_layer=norm_layer)
        unet_block = UnetSkipConnectionVectorBlock(ngf * 2, ngf * 4, input_nc=None, submodule=unet_block,
                                                   norm_layer=norm_layer)
        unet_block = UnetSkipConnectionVectorBlock(ngf, ngf * 2, input_nc=None, submodule=unet_block,
                                                   norm_layer=norm_layer)
        self.model = UnetSkipConnectionVectorBlock(output_nc, ngf, input_nc=input_nc, submodule=unet_block,
                                                   outermost=True, norm_layer=norm_layer,
                                                   last_layer=last_layer)  # add the outermost layer

    def forward(self, input):
        """Standard forward"""
        output = self.model(input)
        vector = self.innermost_block.down_vector_output
        return output, vector


class UnetSkipConnectionVectorBlock(nn.Module):
    """Defines the Unet submodule with skip connection.
        X -------------------identity----------------------
        |-- downsampling -- |submodule| -- upsampling --|
    """

    def __init__(self, outer_nc, inner_nc, input_nc=None,
                 submodule=None, outermost=False, innermost=False, norm_layer=nn.BatchNorm2d, use_dropout=False,
                 inner_conv=True, last_layer='Tanh'):
        """Construct a Unet submodule with skip connections.

        Parameters:
            outer_nc (int) -- the number of filters in the outer conv layer
            inner_nc (int) -- the number of filters in the inner conv layer
            input_nc (int) -- the number of channels in input images/features
            submodule (UnetSkipConnectionBlock) -- previously defined submodules
            outermost (bool)    -- if this module is the outermost module
            innermost (bool)    -- if this module is the innermost module
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers.
        """
        super(UnetSkipConnectionVectorBlock, self).__init__()
        self.outermost = outermost
        self.innermost = innermost
        self.inner_conv = inner_conv
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        if input_nc is None:
            input_nc = outer_nc
        downconv = nn.Conv2d(input_nc, inner_nc, kernel_size=4,
                             stride=2, padding=1, bias=use_bias)
        downrelu = nn.LeakyReLU(0.2, True)
        downnorm = norm_layer(inner_nc)
        uprelu = nn.ReLU(True)
        upnorm = norm_layer(outer_nc)

        if outermost:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1)
            down = [downconv]
            last = None
            if last_layer == 'Sigmoid':
                last = nn.Sigmoid()
            elif last_layer == 'Tanh':
                last = nn.Tanh()
            up = [uprelu, upconv, last]
            model = down + [submodule] + up
        elif innermost:
            upconv = nn.ConvTranspose2d(inner_nc, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias)
            down = [downrelu, downconv]
            up = [uprelu, upconv, upnorm]
            model = down
            self.model_up = nn.Sequential(*up)
        else:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias)
            down = [downrelu, downconv, downnorm]
            up = [uprelu, upconv, upnorm]

            if use_dropout:
                model = down + [submodule] + up + [nn.Dropout(0.5)]
            else:
                model = down + [submodule] + up

        self.model = nn.Sequential(*model)

    def forward(self, x):
        if self.outermost:
            return self.model(x)
        elif self.innermost:
            down_vector_output = self.model(x)
            up_output = self.model_up(down_vector_output)
            if self.inner_conv:
                self.down_vector_output = down_vector_output
            else:
                self.down_vector_output = down_vector_output.view(x.shape[0], -1)
            return torch.cat([x, up_output], 1)
        else:  # add skip connections
            return torch.cat([x, self.model(x)], 1)


class NLayerDiscriminator(nn.Module):
    """Defines a PatchGAN discriminator"""

    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d):
        """Construct a PatchGAN discriminator

        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            n_layers (int)  -- the number of conv layers in the discriminator
            norm_layer      -- normalization layer
        """
        super(NLayerDiscriminator, self).__init__()
        if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        kw = 4
        padw = 1
        sequence = [nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw), nn.LeakyReLU(0.2, True)]
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):  # gradually increase the number of filters
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw, bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]

        sequence += [
            nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]  # output 1 channel prediction map
        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        """Standard forward."""
        return self.model(input)


class PixelDiscriminator(nn.Module):
    """Defines a 1x1 PatchGAN discriminator (pixelGAN)"""

    def __init__(self, input_nc, ndf=64, norm_layer=nn.BatchNorm2d):
        """Construct a 1x1 PatchGAN discriminator

        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            norm_layer      -- normalization layer
        """
        super(PixelDiscriminator, self).__init__()
        if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        self.net = [
            nn.Conv2d(input_nc, ndf, kernel_size=1, stride=1, padding=0),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf, ndf * 2, kernel_size=1, stride=1, padding=0, bias=use_bias),
            norm_layer(ndf * 2),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf * 2, 1, kernel_size=1, stride=1, padding=0, bias=use_bias)]

        self.net = nn.Sequential(*self.net)

    def forward(self, x):
        """Standard forward."""
        return self.net(x)


class ConvDiscriminator(nn.Module):
    """
    使用自带的patchGAN判别器尝试
    """

    def __init__(self, input_nc, ndf=256, norm_layer=nn.BatchNorm2d):
        super(ConvDiscriminator, self).__init__()
        if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        self.net = [
            nn.Conv2d(input_nc, ndf, kernel_size=1, stride=1, padding=0),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf, ndf // 2, kernel_size=1, stride=1, padding=0, bias=use_bias),
            norm_layer(ndf // 2),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf // 2, 1, kernel_size=1, stride=1, padding=0, bias=use_bias)]

        self.net = nn.Sequential(*self.net)

    def forward(self, x):
        """Standard forward."""
        return self.net(x)


class FCDiscriminator(nn.Module):
    """
    使用自带的patchGAN判别器尝试
    """

    def __init__(self, input_nc, ndf=256):
        super(FCDiscriminator, self).__init__()

        self.net = [
            nn.Linear(input_nc, ndf),
            nn.LeakyReLU(0.2, True),
            nn.Linear(ndf, ndf // 2),
            nn.LeakyReLU(0.2, True),
            nn.Linear(ndf // 2, 1)]

        self.net = nn.Sequential(*self.net)

    def forward(self, x):
        """Standard forward."""
        return self.net(x)


class Guide_Block(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, eps_list=None):
        super(Guide_Block, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.eps_list = [0.005, 0.003, 0.0001] if eps_list is None else eps_list
        self.guide_filter_list = []
        for i in eps_list:
            self.guide_filter_list.append(FastGuidedFilter(eps=eps_list[i]))

    def forward(self, x):
        assert x.shape[1] == self.in_channels
        assert len(self.eps_list) == self.out_channels


class GridAttentionBlock(nn.Module):
    def __init__(self, in_channels):
        super(GridAttentionBlock, self).__init__()

        self.inter_channels = in_channels
        self.in_channels = in_channels
        self.gating_channels = in_channels

        self.theta = nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels,
                               kernel_size=1)

        self.phi = nn.Conv2d(in_channels=self.gating_channels, out_channels=self.inter_channels,
                             kernel_size=1, stride=1, padding=0, bias=True)
        self.psi = nn.Conv2d(in_channels=self.inter_channels, out_channels=1, kernel_size=1, stride=1, padding=0,
                             bias=True)

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, g):
        input_size = x.size()
        batch_size = input_size[0]
        assert batch_size == g.size(0)

        theta_x = self.theta(x)
        theta_x_size = theta_x.size()

        phi_g = F.upsample(self.phi(g), size=theta_x_size[2:], mode='bilinear')
        f = F.relu(theta_x + phi_g, inplace=True)

        sigm_psi_f = F.Sigmoid(self.psi(f))

        return sigm_psi_f


class UnetMultiOutputGenerator(nn.Module):
    """Create a Unet-based generator"""

    def __init__(self, input_nc, output_nc, num_downs, output_num=2, ngf=64, norm_layer=nn.BatchNorm2d,
                 use_dropout=False):
        """Construct a Unet generator
        Parameters:
            input_nc (int)  -- the number of channels in input images
            output_nc (int) -- the number of channels in output images
            num_downs (int) -- the number of downsamplings in UNet. For example, # if |num_downs| == 7,
                                image of size 128x128 will become of size 1x1 # at the bottleneck
            ngf (int)       -- the number of filters in the last conv layer
            norm_layer      -- normalization layer

        We construct the U-Net from the innermost layer to the outermost layer.
        It is a recursive process.
        """
        super(UnetMultiOutputGenerator, self).__init__()
        # construct unet structure
        unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, output_num=output_num, input_nc=None, submodule=None,
                                             norm_layer=norm_layer, innermost=True)  # add the innermost layer
        for i in range(num_downs - 5):  # add intermediate layers with ngf * 8 filters
            unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=unet_block,
                                                 norm_layer=norm_layer, use_dropout=use_dropout)
        # gradually reduce the number of filters from ngf * 8 to ngf
        unet_block = UnetSkipConnectionBlock(ngf * 4, ngf * 8, input_nc=None, submodule=unet_block,
                                             norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(ngf * 2, ngf * 4, input_nc=None, submodule=unet_block,
                                             norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(ngf, ngf * 2, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        self.model = UnetSkipConnectionBlock(output_nc, ngf, input_nc=input_nc, submodule=unet_block, outermost=True,
                                             norm_layer=norm_layer)  # add the outermost layer

    def forward(self, input):
        """Standard forward"""
        return self.model(input)


class UnetMultiOutputBlock(nn.Module):
    """Defines the Unet submodule with skip connection.
        X -------------------identity----------------------
        |-- downsampling -- |submodule| -- upsampling --|
    """

    def __init__(self, outer_nc, inner_nc, output_num=2, input_nc=None,
                 submodule=None, outermost=False, innermost=False, norm_layer=nn.BatchNorm2d, use_dropout=False):
        """Construct a Unet submodule with skip connections.

        Parameters:
            outer_nc (int) -- the number of filters in the outer conv layer
            inner_nc (int) -- the number of filters in the inner conv layer
            input_nc (int) -- the number of channels in input images/features
            submodule (UnetSkipConnectionBlock) -- previously defined submodules
            outermost (bool)    -- if this module is the outermost module
            innermost (bool)    -- if this module is the innermost module
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers.
        """
        super(UnetMultiOutputBlock, self).__init__()
        self.outermost = outermost
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        if input_nc is None:
            input_nc = outer_nc
        downconv = nn.Conv2d(input_nc, inner_nc, kernel_size=4,
                             stride=2, padding=1, bias=use_bias)
        downrelu = nn.LeakyReLU(0.2, True)
        downnorm = norm_layer(inner_nc)
        uprelu = nn.ReLU(True)
        upnorm = norm_layer(outer_nc)

        if outermost:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1)
            down = [downconv]
            up = [uprelu, upconv, nn.Tanh()]
            model = down + [submodule] + up
        elif innermost:
            upconv = nn.ConvTranspose2d(inner_nc, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias)
            down = [downrelu, downconv]
            up = [uprelu, upconv, upnorm]
            model = down + up
        else:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias)
            down = [downrelu, downconv, downnorm]
            up = [uprelu, upconv, upnorm]

            if use_dropout:
                model = down + [submodule] + up + [nn.Dropout(0.5)]
            else:
                model = down + [submodule] + up

        self.model = nn.Sequential(*model)

    def forward(self, x):
        if self.outermost:
            return self.model(x)
        else:  # add skip connections
            return torch.cat([x, self.model(x)], 1)


class MultiConv(nn.Module):
    def __init__(self, input_nc, output_nc, conv_block_num=2, kernel_size=3, use_bias=False, norm_func=nn.BatchNorm2d,
                 activation_func=nn.ReLU, use_dropout=False):
        super(MultiConv, self).__init__()
        assert conv_block_num >= 1
        main_conv_list = [nn.Conv2d(input_nc, output_nc, kernel_size=kernel_size, stride=1, padding=1, bias=use_bias),
                          norm_func(output_nc), activation_func(inplace=True)] + (
                             [nn.Dropout(0.5)] if use_dropout else [])

        for _ in range(conv_block_num - 1):
            main_conv_list.append(
                nn.Conv2d(output_nc, output_nc, kernel_size=kernel_size, stride=1, padding=1, bias=use_bias))
            main_conv_list.append(norm_func(output_nc))
            main_conv_list.append(activation_func(inplace=True))
            if use_dropout:
                main_conv_list.append(nn.Dropout(0.5))
        self.main_conv = nn.Sequential(*main_conv_list)

    def forward(self, x):
        return self.main_conv(x)


class UnetMO(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False,
                 last_layer='Sigmoid', activation_func=nn.ReLU, kernel_size=3, conv_num=2):
        super(UnetMO, self).__init__()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        self.down_down = nn.MaxPool2d(2, 2)
        self.down_conv_1 = MultiConv(input_nc, ngf, conv_num, kernel_size, use_bias, norm_layer, activation_func,
                                     use_dropout)
        self.down_conv_2 = MultiConv(ngf, ngf * 2, conv_num, kernel_size, use_bias, norm_layer, activation_func,
                                     use_dropout)
        self.down_conv_3 = MultiConv(ngf * 2, ngf * 4, conv_num, kernel_size, use_bias, norm_layer, activation_func,
                                     use_dropout)
        self.down_conv_4 = MultiConv(ngf * 4, ngf * 8, conv_num, kernel_size, use_bias, norm_layer, activation_func,
                                     use_dropout)
        self.down_conv_5 = MultiConv(ngf * 8, ngf * 16, conv_num, kernel_size, use_bias, norm_layer, activation_func,
                                     use_dropout)

        self.up_conv_1 = MultiConv(ngf * 2, ngf, conv_num, kernel_size, use_bias, norm_layer, activation_func,
                                   use_dropout)
        self.up_conv_2 = MultiConv(ngf * 4, ngf * 2, conv_num, kernel_size, use_bias, norm_layer, activation_func,
                                   use_dropout)
        self.up_conv_3 = MultiConv(ngf * 8, ngf * 4, conv_num, kernel_size, use_bias, norm_layer, activation_func,
                                   use_dropout)
        self.up_conv_4 = MultiConv(ngf * 16, ngf * 8, conv_num, kernel_size, use_bias, norm_layer, activation_func,
                                   use_dropout)

        self.up_up_1 = nn.ConvTranspose2d(ngf * 2, ngf, kernel_size=2, stride=2, padding=0, bias=use_bias)
        self.up_up_2 = nn.ConvTranspose2d(ngf * 4, ngf * 2, kernel_size=2, stride=2, padding=0, bias=use_bias)
        self.up_up_3 = nn.ConvTranspose2d(ngf * 8, ngf * 4, kernel_size=2, stride=2, padding=0, bias=use_bias)
        self.up_up_4 = nn.ConvTranspose2d(ngf * 16, ngf * 8, kernel_size=2, stride=2, padding=0,
                                          bias=use_bias)

        self.h_up_conv_1 = MultiConv(ngf * 2, ngf, conv_num, kernel_size, use_bias, norm_layer, activation_func,
                                     use_dropout)
        self.h_up_conv_2 = MultiConv(ngf * 4, ngf * 2, conv_num, kernel_size, use_bias, norm_layer, activation_func,
                                     use_dropout)
        self.h_up_conv_3 = MultiConv(ngf * 8, ngf * 4, conv_num, kernel_size, use_bias, norm_layer, activation_func,
                                     use_dropout)
        self.h_up_conv_4 = MultiConv(ngf * 16, ngf * 8, conv_num, kernel_size, use_bias, norm_layer, activation_func,
                                     use_dropout)

        self.h_up_up_1 = nn.ConvTranspose2d(ngf * 2, ngf, kernel_size=2, stride=2, padding=0, bias=use_bias)
        self.h_up_up_2 = nn.ConvTranspose2d(ngf * 4, ngf * 2, kernel_size=2, stride=2, padding=0,
                                            bias=use_bias)
        self.h_up_up_3 = nn.ConvTranspose2d(ngf * 8, ngf * 4, kernel_size=2, stride=2, padding=0,
                                            bias=use_bias)
        self.h_up_up_4 = nn.ConvTranspose2d(ngf * 16, ngf * 8, kernel_size=2, stride=2, padding=0,
                                            bias=use_bias)

        self.out = nn.Sequential(
            nn.Conv2d(ngf, output_nc, kernel_size=1, padding=0),
            getattr(torch.nn, last_layer)()
        )

        self.h_out = nn.Sequential(
            nn.Conv2d(ngf, 3, kernel_size=1, padding=0),
            nn.Tanh()
        )

    def forward(self, x):
        x1 = self.down_conv_1(x)
        x2 = self.down_conv_2(self.down_down(x1))
        x3 = self.down_conv_3(self.down_down(x2))
        x4 = self.down_conv_4(self.down_down(x3))
        x5 = self.down_conv_5(self.down_down(x4))

        h_u4 = torch.cat([x4, self.h_up_up_4(x5)], dim=1)
        h_y4 = self.h_up_conv_4(h_u4)
        h_u3 = torch.cat([x3, self.h_up_up_3(h_y4)], dim=1)
        h_y3 = self.h_up_conv_3(h_u3)
        h_u2 = torch.cat([x2, self.h_up_up_2(h_y3)], dim=1)
        h_y2 = self.h_up_conv_2(h_u2)
        h_u1 = torch.cat([x1, self.h_up_up_1(h_y2)], dim=1)
        h_y1 = self.h_up_conv_1(h_u1)

        u4 = torch.cat([h_y4, self.up_up_4(x5)], dim=1)
        y4 = self.up_conv_4(u4)
        u3 = torch.cat([h_y3, self.up_up_3(y4)], dim=1)
        y3 = self.up_conv_3(u3)
        u2 = torch.cat([h_y2, self.up_up_2(y3)], dim=1)
        y2 = self.up_conv_2(u2)
        u1 = torch.cat([h_y1, self.up_up_1(y2)], dim=1)
        y1 = self.up_conv_1(u1)

        o_h = self.h_out(h_y1)
        o = self.out(y1)

        return o_h, o


class SingleUnet(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, last_layer='Sigmoid',
                 activation_func=nn.ReLU, kernel_size=3, conv_num=2, use_bias=False, all_output=False):
        super(SingleUnet, self).__init__()

        self.all_output = all_output

        self.down_down = nn.MaxPool2d(2, 2)
        self.down_conv_1 = MultiConv(input_nc, ngf, conv_num, kernel_size, use_bias, norm_layer, activation_func,
                                     use_dropout)
        self.down_conv_2 = MultiConv(ngf, ngf * 2, conv_num, kernel_size, use_bias, norm_layer, activation_func,
                                     use_dropout)
        self.down_conv_3 = MultiConv(ngf * 2, ngf * 4, conv_num, kernel_size, use_bias, norm_layer, activation_func,
                                     use_dropout)
        self.down_conv_4 = MultiConv(ngf * 4, ngf * 8, conv_num, kernel_size, use_bias, norm_layer, activation_func,
                                     use_dropout)
        self.down_conv_5 = MultiConv(ngf * 8, ngf * 16, conv_num, kernel_size, use_bias, norm_layer, activation_func,
                                     use_dropout)

        self.up_conv_1 = MultiConv(ngf * 2, ngf, conv_num, kernel_size, use_bias, norm_layer, activation_func,
                                   use_dropout)
        self.up_conv_2 = MultiConv(ngf * 4, ngf * 2, conv_num, kernel_size, use_bias, norm_layer, activation_func,
                                   use_dropout)
        self.up_conv_3 = MultiConv(ngf * 8, ngf * 4, conv_num, kernel_size, use_bias, norm_layer, activation_func,
                                   use_dropout)
        self.up_conv_4 = MultiConv(ngf * 16, ngf * 8, conv_num, kernel_size, use_bias, norm_layer, activation_func,
                                   use_dropout)

        self.up_up_1 = nn.ConvTranspose2d(ngf * 2, ngf, kernel_size=2, stride=2, padding=0, bias=use_bias)
        self.up_up_2 = nn.ConvTranspose2d(ngf * 4, ngf * 2, kernel_size=2, stride=2, padding=0, bias=use_bias)
        self.up_up_3 = nn.ConvTranspose2d(ngf * 8, ngf * 4, kernel_size=2, stride=2, padding=0, bias=use_bias)
        self.up_up_4 = nn.ConvTranspose2d(ngf * 16, ngf * 8, kernel_size=2, stride=2, padding=0,
                                          bias=use_bias)

        self.out = nn.Sequential(
            nn.Conv2d(ngf, output_nc, kernel_size=1, padding=0),
            getattr(torch.nn, last_layer)()
        )

    def forward(self, x):
        x1 = self.down_conv_1(x)
        x2 = self.down_conv_2(self.down_down(x1))
        x3 = self.down_conv_3(self.down_down(x2))
        x4 = self.down_conv_4(self.down_down(x3))
        x5 = self.down_conv_5(self.down_down(x4))

        u4 = torch.cat([x4, self.up_up_4(x5)], dim=1)
        y4 = self.up_conv_4(u4)
        u3 = torch.cat([x3, self.up_up_3(y4)], dim=1)
        y3 = self.up_conv_3(u3)
        u2 = torch.cat([x2, self.up_up_2(y3)], dim=1)
        y2 = self.up_conv_2(u2)
        u1 = torch.cat([x1, self.up_up_1(y2)], dim=1)
        y1 = self.up_conv_1(u1)

        o = self.out(y1)

        if self.all_output:
            return o, y1, x1, y2, y3, y4
        return o, y1, x1


class UnetCascade(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False,
                 last_layer='Sigmoid', activation_func=nn.ReLU, kernel_size=3, conv_num=2, original_dense=False):
        super(UnetCascade, self).__init__()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        self.original_dense = original_dense

        self.unet_1 = SingleUnet(input_nc, 3, ngf, norm_layer, use_dropout, 'Tanh', activation_func,
                                 kernel_size, conv_num, use_bias)
        self.unet_2 = SingleUnet(ngf * 2 if original_dense else ngf, output_nc, ngf, norm_layer, use_dropout,
                                 last_layer, activation_func, kernel_size, conv_num, use_bias)

    def forward(self, x):
        h_o, y1, x1 = self.unet_1(x)
        o, _, _ = self.unet_2(torch.cat([x1, y1], dim=1) if self.original_dense else y1)

        return h_o, o


class NaiveAutoEncoder(nn.Module):
    def __init__(self, ngf_list=(64, 128, 256, 512, 1024, 2048)):
        super(NaiveAutoEncoder, self).__init__()

        self.down_down = nn.MaxPool2d(2, 2)

        encoder_list = [MultiConv(3, ngf_list[0], 1)]
        decoder_list = []
        for i in range(len(ngf_list) - 1):
            # print('en', ngf_list[i], ngf_list[i + 1])
            # print('de', ngf_list[-i - 1], ngf_list[-i - 2])
            encoder_list.append(MultiConv(ngf_list[i], ngf_list[i + 1], 2))
            encoder_list.append(nn.MaxPool2d(2, 2))
            decoder_list.append(
                nn.ConvTranspose2d(ngf_list[-i - 1], ngf_list[-i - 1], kernel_size=2, stride=2, padding=0))
            decoder_list.append(MultiConv(ngf_list[-i - 1], ngf_list[-i - 2]))
        encoder_list.append(MultiConv(ngf_list[-1], ngf_list[-1]))
        decoder_list.append(nn.Conv2d(ngf_list[0], 3, 1))
        self.encoder = nn.Sequential(*encoder_list)
        self.decoder = nn.Sequential(*decoder_list)

    def forward(self, x):
        latent_vector = self.encoder(x)
        result = torch.nn.Tanh()(self.decoder(latent_vector))

        return latent_vector, result

