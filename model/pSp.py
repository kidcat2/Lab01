import torch
import torch.nn            as nn
import torch.nn.functional as F
import numpy               as np
import torchvision.models as models
import math

def get_block(in_channel, depth, num_units, stride=2):
	return [Resblock(in_channel, depth, stride)] + [Resblock(depth, depth, 1) for i in range(num_units - 1)]

def get_blocks():
    blocks = [
        get_block(in_channel=64, depth=64, num_units=3),
        get_block(in_channel=64, depth=128, num_units=4),
        get_block(in_channel=128, depth=256, num_units=4), # 14
        get_block(in_channel=256, depth=512, num_units=3)
    ]
    return blocks

# input layer : 64 x 256 x 256  
# layer1 : 64 x 128 x 128 -- 1 + 2  (0~2)
# layer2 : 128 x 64 x 64 --  1 + 3  (3~6)
# layer3 : 256 x 32 x 32 --  1 + 13 (7~20)--> 1 + 3 (7~10)
# layer4 : 512 x 16 x 16 --  1 + 2 (21~23) --> (11~13)

def input_layer(in_channel):
    return nn.Sequential(
        nn.Conv2d(in_channel, 64, (3, 3), stride=1, padding=1, bias=False),
        nn.BatchNorm2d(64),
        nn.PReLU(64)
    )

class EqualLinear(nn.Module): # Delete Fused Leaky ReLu (cuda coding)
    def __init__(
            self, in_dim, out_dim, bias=True, bias_init=0, lr_mul=1, activation=None
    ):
        super().__init__()

        self.weight = nn.Parameter(torch.randn(out_dim, in_dim).div_(lr_mul))

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_dim).fill_(bias_init))

        else:
            self.bias = None

        self.activation = activation

        self.scale = (1 / math.sqrt(in_dim)) * lr_mul
        self.lr_mul = lr_mul

    def forward(self, input):

        """ fused_leaky_relu codes --

        if self.activation:
            out = F.linear(input, self.weight * self.scale)
            out = fused_leaky_relu(out, self.bias * self.lr_mul)

        else:
            out = F.linear(
                input, self.weight * self.scale, bias=self.bias * self.lr_mul
            )

        """ 
        out = F.linear(input, self.weight * self.scale, bias=self.bias * self.lr_mul)
        if self.activation:
            out = F.leaky_relu(out)

        return out

    def __repr__(self):
        return (
            f'{self.__class__.__name__}({self.weight.shape[1]}, {self.weight.shape[0]})'
        )

class GradualStyleBlock(nn.Module):
    def __init__(self, in_c, out_c, spatial):
        super(GradualStyleBlock, self).__init__()
        self.out_c = out_c
        self.spatial = spatial
        num_pools = int(np.log2(spatial))
        modules = []
        modules += [nn.Conv2d(in_c, out_c, kernel_size=3, stride=2, padding=1),
                    nn.LeakyReLU()]
        for i in range(num_pools - 1):
            modules += [
                nn.Conv2d(out_c, out_c, kernel_size=3, stride=2, padding=1),
                nn.LeakyReLU()
            ]
        self.convs = nn.Sequential(*modules)
        self.linear = EqualLinear(out_c, out_c, lr_mul=1)

    def forward(self, x):
        x = self.convs(x)
        x = x.view(-1, self.out_c)
        x = self.linear(x)
        return x



class SEModule(nn.Module):
	def __init__(self, channels, reduction):
		super(SEModule, self).__init__()
		self.avg_pool = nn.AdaptiveAvgPool2d(1)
		self.fc1 = nn.Conv2d(channels, channels // reduction, kernel_size=1, padding=0, bias=False)
		self.relu = nn.ReLU(inplace=True)
		self.fc2 = nn.Conv2d(channels // reduction, channels, kernel_size=1, padding=0, bias=False)
		self.sigmoid = nn.Sigmoid()

	def forward(self, x):
		module_input = x
		x = self.avg_pool(x)
		x = self.fc1(x)
		x = self.relu(x)
		x = self.fc2(x)
		x = self.sigmoid(x)
		return module_input * x

class Resblock(nn.Module):
    def __init__(self, in_channel, depth, stride):
        super(Resblock, self).__init__()
        if in_channel == depth:
            self.shortcut_layer = nn.MaxPool2d(1, stride)
        else:
            self.shortcut_layer = nn.Sequential(
                nn.Conv2d(in_channel, depth, (1, 1), stride, bias=False),
                nn.BatchNorm2d(depth)
            )
        self.res_layer = nn.Sequential(
            nn.BatchNorm2d(in_channel),
            nn.Conv2d(in_channel, depth, (3, 3), (1, 1), 1, bias=False),
            nn.PReLU(depth),
            nn.Conv2d(depth, depth, (3, 3), stride, 1, bias=False),
            nn.BatchNorm2d(depth),
            SEModule(depth, 16)
        )

    def forward(self, x):
        shortcut = self.shortcut_layer(x)
        res = self.res_layer(x)
        return res + shortcut

class pSpEncoder(nn.Module):
    def __init__(self, in_channel = 3):
        super(pSpEncoder, self).__init__()
        blocks = get_blocks()
        self.input_layer = input_layer(in_channel)
    
        modules = []
        for block in blocks:
            for resblock in block:
                modules.append(resblock)

        self.body = nn.Sequential(*modules)

        self.styles = nn.ModuleList()
        self.coarse = 1
        self.middle = 2
        self.fine = 3

        for i in range(self.fine):
            if i < self.coarse:
                style = GradualStyleBlock(512, 512, 16)
            elif i < self.middle:
                style = GradualStyleBlock(512, 512, 32)
            else:
                style = GradualStyleBlock(512, 512, 64)
            self.styles.append(style)

        self.latlayer1 = nn.Conv2d(256, 512, kernel_size=1, stride=1, padding=0)
        self.latlayer2 = nn.Conv2d(128, 512, kernel_size=1, stride=1, padding=0)

    def _upsample_add(self, x, y):
        _, _, H, W = y.size()
        return F.interpolate(x, size=(H, W), mode='bilinear', align_corners=True) + y
    
    def forward(self, x): # x : b, img개수, 3, 256, 256
        B, N, _, _, _ = x.size()
        x = x.view(-1, 3, 256, 256) # b*img개수, 3, 256, 256
        x = self.input_layer(x)

        latents = []
        modulelist = list(self.body._modules.values())
        for i, l in enumerate(modulelist):
            x = l(x)
            if i == 6:
                c1 = x # [b*img_cnt, 128, 64, 64]
            elif i == 10:
                c2 = x # [b*img_cnt, 256, 32, 32]
            elif i == 13:
                c3 = x # [b*img_cnt, 512, 16, 16]

        latents.append(c1)
        latents.append(c1)
        latents.append(c2)
        latents.append(c2)
        latents.append(c3)
        
        return latents
        for j in range(self.coarse):
            latents.append(self.styles[j](c3))

        p2 = self._upsample_add(c3, self.latlayer1(c2))
        for j in range(self.coarse, self.middle):
            latents.append(self.styles[j](p2))

        p1 = self._upsample_add(p2, self.latlayer2(c1))
        for j in range(self.middle, self.fine ):
            latents.append(self.styles[j](p1))

        out = torch.stack(latents, dim = 1) # batch*img_cnt, style_cnt, 512
        out = out.view(B, N, -1, 512) # batch, img_cnt, style_cnt, 512
        out = torch.mean(out, dim = 1)
        return out
    

""" Fuse Leaky ReLu ------------------------
class FusedLeakyReLUFunctionBackward(Function):
    @staticmethod
    def forward(ctx, grad_output, out, negative_slope, scale):
        ctx.save_for_backward(out)
        ctx.negative_slope = negative_slope
        ctx.scale = scale

        empty = grad_output.new_empty(0)

        grad_input = fused.fused_bias_act(
            grad_output, empty, out, 3, 1, negative_slope, scale
        )

        dim = [0]

        if grad_input.ndim > 2:
            dim += list(range(2, grad_input.ndim))

        grad_bias = grad_input.sum(dim).detach()

        return grad_input, grad_bias

    @staticmethod
    def backward(ctx, gradgrad_input, gradgrad_bias):
        out, = ctx.saved_tensors
        gradgrad_out = fused.fused_bias_act(
            gradgrad_input, gradgrad_bias, out, 3, 1, ctx.negative_slope, ctx.scale
        )

        return gradgrad_out, None, None, None


class FusedLeakyReLUFunction(Function):
    @staticmethod
    def forward(ctx, input, bias, negative_slope, scale):
        empty = input.new_empty(0)
        out = fused.fused_bias_act(input, bias, empty, 3, 0, negative_slope, scale)
        ctx.save_for_backward(out)
        ctx.negative_slope = negative_slope
        ctx.scale = scale

        return out

    @staticmethod
    def backward(ctx, grad_output):
        out, = ctx.saved_tensors

        grad_input, grad_bias = FusedLeakyReLUFunctionBackward.apply(
            grad_output, out, ctx.negative_slope, ctx.scale
        )

        return grad_input, grad_bias, None, None


class FusedLeakyReLU(nn.Module):
    def __init__(self, channel, negative_slope=0.2, scale=2 ** 0.5):
        super().__init__()

        self.bias = nn.Parameter(torch.zeros(channel))
        self.negative_slope = negative_slope
        self.scale = scale

    def forward(self, input):
        return fused_leaky_relu(input, self.bias, self.negative_slope, self.scale)


def fused_leaky_relu(input, bias, negative_slope=0.2, scale=2 ** 0.5):
    return FusedLeakyReLUFunction.apply(input, bias, negative_slope, scale)

"""
