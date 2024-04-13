import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.network_utils import get_embedder
from utils.ref_utils import components_from_spherical_harmonics

class IdentityActivation(nn.Module):
    def forward(self, x): return x

class ExpActivation(nn.Module):
    def __init__(self, max_light=5.0):
        super().__init__()
        self.max_light=max_light

    def forward(self, x):
        return torch.exp(torch.clamp(x, max=self.max_light))

def make_predictor_2layer(feats_dim: object, output_dim: object, weight_norm: object = True, activation='sigmoid', exp_max=0.0, run_dim=128) -> object:
    if activation == 'sigmoid':
        activation = nn.Sigmoid()
    elif activation=='exp':
        activation = ExpActivation(max_light=exp_max)
    elif activation=='none':
        activation = IdentityActivation()
    elif activation=='relu':
        activation = nn.ReLU()
    elif activation=='softplus':
        activation = nn.Softplus()
    else:
        raise NotImplementedError

    if weight_norm:
        module=nn.Sequential(
            nn.utils.weight_norm(nn.Linear(feats_dim, run_dim)),
            nn.ReLU(),
            nn.utils.weight_norm(nn.Linear(run_dim, output_dim)),
            activation,
        )
    else:
        module=nn.Sequential(
            nn.Linear(feats_dim, run_dim),
            nn.ReLU(),
            nn.Linear(run_dim, output_dim),
            activation,
        )
    return module
        
def make_predictor_3layer(feats_dim: object, output_dim: object, weight_norm: object = True, activation='sigmoid', exp_max=0.0, run_dim=128) -> object:
    if activation == 'sigmoid':
        activation = nn.Sigmoid()
    elif activation=='exp':
        activation = ExpActivation(max_light=exp_max)
    elif activation=='none':
        activation = IdentityActivation()
    elif activation=='relu':
        activation = nn.ReLU()
    elif activation=='softplus':
        activation = nn.Softplus()
    elif activation=='tanh':
        activation = nn.Tanh()
    else:
        raise NotImplementedError

    if weight_norm:
        module=nn.Sequential(
            nn.utils.weight_norm(nn.Linear(feats_dim, run_dim)),
            nn.ReLU(),
            nn.utils.weight_norm(nn.Linear(run_dim, run_dim)),
            nn.ReLU(),
            nn.utils.weight_norm(nn.Linear(run_dim, output_dim)),
            activation,
        )
    else:
        module=nn.Sequential(
            nn.Linear(feats_dim, run_dim),
            nn.ReLU(),
            nn.Linear(run_dim, run_dim),
            nn.ReLU(),
            nn.Linear(run_dim, output_dim),
            activation,
        )
    return module

def make_predictor_4layer(feats_dim: object, output_dim: object, weight_norm: object = True, activation='sigmoid', exp_max=0.0, run_dim = 256) -> object:
    if activation == 'sigmoid':
        activation = nn.Sigmoid()
    elif activation=='exp':
        activation = ExpActivation(max_light=exp_max)
    elif activation=='none':
        activation = IdentityActivation()
    elif activation=='relu':
        activation = nn.ReLU()
    else:
        raise NotImplementedError

    if weight_norm:
        module=nn.Sequential(
            nn.utils.weight_norm(nn.Linear(feats_dim, run_dim)),
            nn.ReLU(),
            nn.utils.weight_norm(nn.Linear(run_dim, run_dim)),
            nn.ReLU(),
            nn.utils.weight_norm(nn.Linear(run_dim, run_dim)),
            nn.ReLU(),
            nn.utils.weight_norm(nn.Linear(run_dim, output_dim)),
            activation,
        )
    else:
        module=nn.Sequential(
            nn.Linear(feats_dim, run_dim),
            nn.ReLU(),
            nn.Linear(run_dim, run_dim),
            nn.ReLU(),
            nn.Linear(run_dim, run_dim),
            nn.ReLU(),
            nn.Linear(run_dim, output_dim),
            activation,
        )

    return module

def gaussian_kernel2D(kernel_size=5, sigma=1.0):
    x = torch.arange(-kernel_size // 2 + 1.0, kernel_size // 2 + 1.0)
    y = torch.arange(-kernel_size // 2 + 1.0, kernel_size // 2 + 1.0)
    xx, yy = torch.meshgrid(x, y)
    kernel = torch.exp(-(xx**2 + yy**2) / (2 * sigma**2))
    kernel = (kernel[None, None, ...] / kernel.sum())
    return kernel

def gaussian_kernel1D(kernel_size=5, sigma=1.0):
    x = torch.arange(-kernel_size // 2 + 1.0, kernel_size // 2 + 1.0)
    kernel = torch.exp(-x**2 / (2 * sigma**2))
    kernel = (kernel[None, None, ...] / kernel.sum())
    return kernel     
    
class SHEncoding(nn.Module):
    def __init__(self, levels):
        super().__init__()
        if levels <= 0 or levels > 4:
            raise ValueError(f"Spherical harmonic encoding only supports 1 to 4 levels, requested {levels}")
        self.levels = levels        
        self.out_dims = self.levels**2
        
    def forward(self, inputs):
        return components_from_spherical_harmonics(self.levels, inputs)
        
class GaussianBlur2D(nn.Module):
    def __init__(self, kernel_size=5, sigma=1.0, stride=2, device='cuda'):
        super(GaussianBlur2D, self).__init__()
        self.kernel_size = kernel_size
        self.sigma = sigma
        self.stride = stride
        self.register_buffer('kernel', gaussian_kernel2D(kernel_size, sigma).to(device))
        
    def forward(self, x):
        return F.conv2d(x, self.kernel, stride=self.stride, padding=self.kernel_size // 2)

class GaussianBlur1D(nn.Module):
    def __init__(self, kernel_size=5, sigma=1.0, stride=2, device='cuda'):
        super(GaussianBlur1D, self).__init__()
        self.kernel_size = kernel_size
        self.sigma = sigma
        self.stride = stride
        self.register_buffer('kernel', gaussian_kernel1D(kernel_size, sigma).to(device))
        
    def forward(self, x):
        return F.conv1d(x, self.kernel, stride=self.stride, padding=self.kernel_size // 2)

class TVLoss(nn.Module):
    def __init__(self,TVLoss_weight=1):
        super(TVLoss,self).__init__()
        self.TVLoss_weight = TVLoss_weight

    def forward(self,x):
        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]
        count_h = self._tensor_size(x[:,:,1:,:])
        count_w = self._tensor_size(x[:,:,:,1:])
        total_loss = 0.
        if count_h != 0:
            h_tv = torch.pow((x[:,:,1:,:]-x[:,:,:h_x-1,:]),2).sum()
            total_loss += h_tv/count_h
        if count_w != 0:
            w_tv = torch.pow((x[:,:,:,1:]-x[:,:,:,:w_x-1]),2).sum()
            total_loss += w_tv/count_w
        return self.TVLoss_weight*2*(total_loss)/batch_size

    def _tensor_size(self,t):
        return t.size()[1]*t.size()[2]*t.size()[3]

class SingleVarianceNetwork(nn.Module):
    def __init__(self, init_val, activation='exp'):
        super(SingleVarianceNetwork, self).__init__()
        self.act = activation
        self.register_parameter('variance', nn.Parameter(torch.tensor(init_val)))

    def forward(self, x):
        if self.act=='exp':
            return torch.ones([*x.shape[:-1], 1]) * torch.exp(self.variance * 10.0)
        elif self.act=='linear':
            return torch.ones([*x.shape[:-1], 1]) * self.variance * 10.0
        elif self.act=='square':
            return torch.ones([*x.shape[:-1], 1]) * (self.variance * 10.0) ** 2
        else:
            raise NotImplementedError

    def warp(self, x, inv_s):
        return torch.ones([*x.shape[:-1], 1]) * inv_s

# This implementation is borrowed from nerf-pytorch: https://github.com/yenchenlin/nerf-pytorch
class NeRFNetwork(nn.Module):
    def __init__(self,
                 D=8,
                 W=256,
                 d_in=3,
                 d_in_view=3,
                 multires=0,
                 multires_view=0,
                 output_ch=4,
                 skips=[4],
                 use_viewdirs=False):
        super(NeRFNetwork, self).__init__()
        self.D = D
        self.W = W
        self.d_in = d_in
        self.d_in_view = d_in_view
        self.input_ch = 3
        self.input_ch_view = 3
        self.embed_fn = None
        self.embed_fn_view = None

        if multires > 0:
            embed_fn, input_ch = get_embedder(multires, input_dims=d_in)
            self.embed_fn = embed_fn
            self.input_ch = input_ch

        if multires_view > 0:
            embed_fn_view, input_ch_view = get_embedder(multires_view, input_dims=d_in_view)
            self.embed_fn_view = embed_fn_view
            self.input_ch_view = input_ch_view

        self.skips = skips
        self.use_viewdirs = use_viewdirs

        self.pts_linears = nn.ModuleList(
            [nn.Linear(self.input_ch, W)] +
            [nn.Linear(W, W) if i not in self.skips else nn.Linear(W + self.input_ch, W) for i in range(D - 1)])

        ### Implementation according to the official code release
        ### (https://github.com/bmild/nerf/blob/master/run_nerf_helpers.py#L104-L105)
        self.views_linears = nn.ModuleList([nn.Linear(self.input_ch_view + W, W // 2)])

        ### Implementation according to the paper
        # self.views_linears = nn.ModuleList(
        #     [nn.Linear(input_ch_views + W, W//2)] + [nn.Linear(W//2, W//2) for i in range(D//2)])

        if use_viewdirs:
            self.feature_linear = nn.Linear(W, W)
            self.alpha_linear = nn.Linear(W, 1)
            self.rgb_linear = nn.Linear(W // 2, 3)
        else:
            self.output_linear = nn.Linear(W, output_ch)

    def forward(self, input_pts, input_views):
        if self.embed_fn is not None:
            input_pts = self.embed_fn(input_pts)
        if self.embed_fn_view is not None:
            input_views = self.embed_fn_view(input_views)

        h = input_pts
        for i, l in enumerate(self.pts_linears):
            h = self.pts_linears[i](h)
            h = F.relu(h)
            if i in self.skips:
                h = torch.cat([input_pts, h], -1)

        if self.use_viewdirs:
            alpha = self.alpha_linear(h)
            feature = self.feature_linear(h)
            h = torch.cat([feature, input_views], -1)

            for i, l in enumerate(self.views_linears):
                h = self.views_linears[i](h)
                h = F.relu(h)

            rgb = self.rgb_linear(h)
            return alpha, rgb
        else:
            assert False

    def density(self, input_pts):
        if self.embed_fn is not None:
            input_pts = self.embed_fn(input_pts)

        h = input_pts
        for i, l in enumerate(self.pts_linears):
            h = self.pts_linears[i](h)
            h = F.relu(h)
            if i in self.skips:
                h = torch.cat([input_pts, h], -1)

        alpha = self.alpha_linear(h)
        return alpha