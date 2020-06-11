import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from MPNCOV.python import MPNCOV


def same_conv(in_channels, out_channels, kernel_size, bias=True):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size//2), bias=bias)


class _NonLocalBlockND(nn.Module):
    def __init__(self, in_channels, inter_channels=None, dimension=3, mode='embedded_gaussian',
                 sub_sample=True, bn_layer=True):
        super().__init__()

        assert dimension in [1, 2, 3]
        assert mode in ['embedded_gaussian', 'gaussian', 'dot_product', 'concatenation']

        #print('Dimension: %d, mode: %s' % (dimension, mode))

        self.mode = mode
        self.dimension = dimension
        self.sub_sample = sub_sample

        self.in_channels = in_channels
        self.inter_channels = inter_channels

        if self.inter_channels is None:
            self.inter_channels = in_channels // 2
            if self.inter_channels == 0:
                self.inter_channels = 1

        if dimension == 3:
            conv_nd = nn.Conv3d
            max_pool = nn.MaxPool3d
            bn = nn.BatchNorm3d
        elif dimension == 2:
            conv_nd = nn.Conv2d
            max_pool = nn.MaxPool2d
            bn = nn.BatchNorm2d
        else:
            conv_nd = nn.Conv1d
            max_pool = nn.MaxPool1d
            bn = nn.BatchNorm1d

        self.g = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                         kernel_size=1, stride=1, padding=0)
        
        if bn_layer:
            self.W = nn.Sequential(
                conv_nd(in_channels=self.inter_channels, out_channels=self.in_channels,
                        kernel_size=1, stride=1, padding=0),
                bn(self.in_channels)
            )
            nn.init.constant_(self.W[1].weight, 0)
            nn.init.constant_(self.W[1].bias, 0)
        else:
            self.W = conv_nd(in_channels=self.inter_channels, out_channels=self.in_channels,
                             kernel_size=1, stride=1, padding=0)
            nn.init.constant_(self.W.weight, 0)
            nn.init.constant_(self.W.bias, 0)

        self.theta = None
        self.phi = None
        self.concat_project = None
        if mode in ['embedded_gaussian', 'dot_product', 'concatenation']:
            self.theta = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                                 kernel_size=1, stride=1, padding=0)
            self.phi = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                               kernel_size=1, stride=1, padding=0)

            if mode == 'embedded_gaussian':
                self.operation_function = self._embedded_gaussian
            elif mode == 'dot_product':
                self.operation_function = self._dot_product
            elif mode == 'concatenation':
                self.operation_function = self._concatenation
                self.concat_project = nn.Sequential(
                    nn.Conv2d(self.inter_channels * 2, 1, 1, 1, 0, bias=False),
                    nn.ReLU()
                )
        elif mode == 'gaussian':
            self.operation_function = self._gaussian

        if sub_sample:
            self.g = nn.Sequential(self.g, max_pool(kernel_size=2))
            if self.phi is None:
                self.phi = max_pool(kernel_size=2)
            else:
                self.phi = nn.Sequential(self.phi, max_pool(kernel_size=2))

    def forward(self, x):
        return self.operation_function(x)

    def _embedded_gaussian(self, x):
        b, _, h, w = x.shape

        g_x = self.g(x).view(b, self.inter_channels, -1)
        g_x = g_x.permute(0, 2, 1)

        theta_x = self.theta(x).view(b, self.inter_channels, -1)
        theta_x = theta_x.permute(0, 2, 1)

        phi_x = self.phi(x).view(b, self.inter_channels, -1)

        f = torch.matmul(theta_x, phi_x)
        f_div_C = F.softmax(f, dim=-1)

        y = torch.matmul(f_div_C, g_x)
        y = y.permute(0, 2, 1).contiguous()
        y = y.view(b, self.inter_channels, h, w)

        W_y = self.W(y)
        z = W_y + x

        return z

    def _gaussian(self, x):
        b, _, h, w = x.shape

        g_x = self.g(x).view(b, self.inter_channels, -1)
        g_x = g_x.permute(0, 2, 1)

        theta_x = x.view(b, self.in_channels, -1)
        theta_x = theta_x.permute(0, 2, 1)

        if self.sub_sample:
            phi_x = self.phi(x).view(b, self.in_channels, -1)
        else:
            phi_x = x.view(b, self.in_channels, -1)

        f = torch.matmul(theta_x, phi_x)
        f_div_C = F.softmax(f, dim=-1)

        y = torch.matmul(f_div_C, g_x)
        y = y.permute(0, 2, 1).contiguous()
        y = y.view(b, self.inter_channels, h, w)

        W_y = self.W(y)
        z = W_y + x

        return z

    def _dot_product(self, x):
        b, _, h, w = x.shape

        g_x = self.g(x).view(b, self.inter_channels, -1)
        g_x = g_x.permute(0, 2, 1)

        theta_x = self.theta(x).view(b, self.inter_channels, -1)
        theta_x = theta_x.permute(0, 2, 1)

        phi_x = self.phi(x).view(b, self.inter_channels, -1)

        f = torch.matmul(theta_x, phi_x)
        N = f.size(-1)
        f_div_C = f / N

        y = torch.matmul(f_div_C, g_x)
        y = y.permute(0, 2, 1).contiguous()
        y = y.view(b, self.inter_channels, h, w)

        W_y = self.W(y)
        z = W_y + x

        return z

    def _concatenation(self, x):
        batch_size = x.size(0)

        g_x = self.g(x).view(batch_size, self.inter_channels, -1)
        g_x = g_x.permute(0, 2, 1)

        theta_x = self.theta(x).view(batch_size, self.inter_channels, -1, 1)

        phi_x = self.phi(x).view(batch_size, self.inter_channels, 1, -1)

        h = theta_x.size(2)
        w = phi_x.size(3)
        theta_x = theta_x.repeat(1, 1, 1, w)
        phi_x = phi_x.repeat(1, 1, h, 1)

        concat_feature = torch.cat([theta_x, phi_x], dim=1)
        f = self.concat_project(concat_feature)
        b, _, h, w = f.size()
        f = f.view(b, h, w)

        N = f.size(-1)
        f_div_C = f / N

        y = torch.matmul(f_div_C, g_x)
        y = y.permute(0, 2, 1).contiguous()
        y = y.view(batch_size, self.inter_channels, *x.size()[2:])

        W_y = self.W(y)
        z = W_y + x

        return z


class NONLocalBlock1D(_NonLocalBlockND):
    def __init__(self, in_channels, inter_channels=None, mode='embedded_gaussian', sub_sample=True, bn_layer=True):
        super().__init__(in_channels,
                         inter_channels=inter_channels,
                         dimension=1, mode=mode,
                         sub_sample=sub_sample,
                         bn_layer=bn_layer)


class NONLocalBlock2D(_NonLocalBlockND):
    def __init__(self, in_channels, inter_channels=None, mode='embedded_gaussian', sub_sample=True, bn_layer=True):
        super().__init__(in_channels,
                         inter_channels=inter_channels,
                         dimension=2, mode=mode,
                         sub_sample=sub_sample,
                         bn_layer=bn_layer)


class SOCA(nn.Module):
    def __init__(self, channel, reduction=8):
        super().__init__()

        self.conv_du = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, h, w = x.shape

        h1 = 1000
        w1 = 1000
        if h < h1 and w < w1:
            x_sub = x
        elif h < h1 and w > w1:
            W = (w - w1) // 2
            x_sub = x[:, :, :, W:(W + w1)]
        elif w < w1 and h > h1:
            H = (h - h1) // 2
            x_sub = x[:, :, H:H + h1, :]
        else:
            H = (h - h1) // 2
            W = (w - w1) // 2
            x_sub = x[:, :, H:(H + h1), W:(W + w1)]

        # MPN-COV
        cov_mat = MPNCOV.CovpoolLayer(x_sub)
        cov_mat_sqrt = MPNCOV.SqrtmLayer(cov_mat, 5)

        cov_mat_sum = torch.mean(cov_mat_sqrt, 1)
        cov_mat_sum = cov_mat_sum.view(b, c, 1, 1)

        y_cov = self.conv_du(cov_mat_sum)

        return y_cov*x


class Nonlocal_CA(nn.Module):
    def __init__(self, in_feat=64, inter_feat=32, sub_sample=False, bn_layer=True):
        super().__init__()

        self.non_local = NONLocalBlock2D(in_feat, inter_channels=inter_feat, sub_sample=sub_sample, bn_layer=bn_layer)

    def forward(self, x):
        _, _, h, w = x.shape

        h1 = int(h / 2)
        w1 = int(w / 2)
        nonlocal_feat = torch.zeros_like(x)

        feat_sub_lu = x[:, :, :h1, :w1]
        feat_sub_ld = x[:, :, h1:, :w1]
        feat_sub_ru = x[:, :, :h1, w1:]
        feat_sub_rd = x[:, :, h1:, w1:]

        nonlocal_lu = self.non_local(feat_sub_lu)
        nonlocal_ld = self.non_local(feat_sub_ld)
        nonlocal_ru = self.non_local(feat_sub_ru)
        nonlocal_rd = self.non_local(feat_sub_rd)
        nonlocal_feat[:, :, :h1, :w1] = nonlocal_lu
        nonlocal_feat[:, :, h1:, :w1] = nonlocal_ld
        nonlocal_feat[:, :, :h1, w1:] = nonlocal_ru
        nonlocal_feat[:, :, h1:, w1:] = nonlocal_rd

        return nonlocal_feat


class RB(nn.Module):
    def __init__(self, conv, n_feat, kernel_size, reduction,
                 bias=True, bn=False, act=nn.ReLU(True), res_scale=1):
        super().__init__()

        self.res_scale = res_scale

        layers = []
        for i in range(2):
            layers.append(conv(n_feat, n_feat, kernel_size, bias=bias))
            if bn: layers.append(nn.BatchNorm2d(n_feat))
            if i == 0: layers.append(act)
        self.body = nn.Sequential(*layers)

    def forward(self, x):
        res = self.body(x)
        #res = self.body(x).mul_(self.res_scale)
        res += x
        return res


class LSRAG(nn.Module):
    def __init__(self, conv, n_feat, kernel_size, reduction, act, res_scale, n_resblocks):
        super().__init__()

        layers = []
        for _ in range(n_resblocks):
            layers.append(RB(conv, n_feat, kernel_size, reduction, bias=True, bn=False, act=act, res_scale=res_scale))
        self.rcab = nn.Sequential(*layers)

        self.soca = SOCA(n_feat, reduction=reduction)
        self.conv_last = conv(n_feat, n_feat, kernel_size)
    
    def forward(self, x):
        res = self.rcab(x)
        res = self.soca(res)
        res = self.conv_last(res)
        res += x
        return res


class SAN(nn.Module):
    def __init__(self, conv=same_conv,  n_colors=3, n_feat=64, kernel_size=3,
                 n_resgroups=10, n_resblocks=20,
                 act=nn.ReLU(True),
                 reduction=16, res_scale=1,
                 scale=2):
        super().__init__()

        layers = [conv(n_colors, n_feat, kernel_size)]
        self.head = nn.Sequential(*layers)

        self.non_local = Nonlocal_CA(n_feat, n_feat // 8, sub_sample=False, bn_layer=False)

        self.gamma = nn.Parameter(torch.zeros(1))
        layers = []
        for _ in range(n_resgroups):
            layers.append(LSRAG(conv, n_feat, kernel_size, reduction, act, res_scale, n_resblocks))
        self.rg = nn.ModuleList(layers)

        layers = []
        if (scale & (scale - 1)) == 0:
            for _ in range(int(math.log(scale, 2))):
                layers.append(conv(n_feat, 4*n_feat, kernel_size))
                layers.append(nn.PixelShuffle(2))
        elif scale == 3:
            layers.append(conv(n_feat, 9*n_feat, kernel_size))
            layers.append(nn.PixelShuffle(3))
        layers.append(conv(n_feat, n_colors, kernel_size))
        self.tail = nn.Sequential(*layers)

    def forward(self, x):
        x = self.head(x)

        xx = self.non_local(x)
        residual = xx
        
        for l in self.rg:
            xx = l(xx) + self.gamma*residual

        res = self.non_local(xx)
        res += x

        x = self.tail(res)

        return x

    def load_state_dict(self, state_dict, strict=False, transfer=False):
        own_state = self.state_dict()
        for name, param in state_dict.items():
            if name in own_state:
                if isinstance(param, nn.Parameter):
                    param = param.data
                try:
                    own_state[name].copy_(param)
                except Exception:
                    if name.find('tail') >= 0 and transfer:
                        print('Replace pre-trained upsampler to new one...')
                    else:
                        raise RuntimeError('While copying the parameter named {}, '
                                           'whose dimensions in the model are {} and '
                                           'whose dimensions in the checkpoint are {}.'
                                           .format(name, own_state[name].size(), param.size()))
            elif strict:
                if name.find('tail') == -1:
                    raise KeyError('unexpected key "{}" in state_dict'.format(name))

        if strict:
            missing = set(own_state.keys()) - set(state_dict.keys())
            if len(missing) > 0:
                raise KeyError('missing keys in state_dict: "{}"'.format(missing))
