"""
If FX.Graph is traced for auto-parallel module, some extra node will be added during
graph construction to deal with the compatibility between bias-addition and all-reduce.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .symbolic_trace import register_tracer_impl


@register_tracer_impl(F.linear, name='_bias_addition_impl')
def linear_impl(input, weight, bias=None):
    if bias is None:
        return F.linear(input, weight)
    else:
        return F.linear(input, weight) + bias


@register_tracer_impl(F.conv1d, name='_bias_addition_impl')
def conv1d_impl(input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1):
    if bias is None:
        return F.conv1d(input, weight, stride=stride, padding=padding, dilation=dilation, groups=groups)
    else:
        return F.conv1d(input, weight, stride=stride, padding=padding, dilation=dilation, groups=groups) + bias


@register_tracer_impl(F.conv2d, name='_bias_addition_impl')
def conv2d_impl(input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1):
    if bias is None:
        return F.conv2d(input, weight, stride=stride, padding=padding, dilation=dilation, groups=groups)
    else:
        return F.conv2d(input, weight, stride=stride, padding=padding, dilation=dilation, groups=groups) + bias


@register_tracer_impl(F.conv3d, name='_bias_addition_impl')
def conv3d_impl(input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1):
    if bias is None:
        return F.conv3d(input, weight, stride=stride, padding=padding, dilation=dilation, groups=groups)
    else:
        return F.conv3d(input, weight, stride=stride, padding=padding, dilation=dilation, groups=groups) + bias


@register_tracer_impl(torch.addmm, name='_bias_addition_impl')
@register_tracer_impl(torch.Tensor.addmm, name='_bias_addition_impl')
def addmm_impl(input, mat1, mat2, beta=None, alpha=None):
    if alpha is not None and beta is not None:
        return F.linear(mat1, mat2.transpose(0, 1)) * alpha + input * beta
    elif alpha is not None:
        return F.linear(mat1, mat2.transpose(0, 1)) * alpha + input
    elif beta is not None:
        return F.linear(mat1, mat2.transpose(0, 1)) + input * beta
    else:
        return F.linear(mat1, mat2.transpose(0, 1)) + input


@register_tracer_impl(torch.addbmm, name='_bias_addition_impl')
@register_tracer_impl(torch.Tensor.addbmm, name='_bias_addition_impl')
def addbmm_impl(input, batch1, batch2, beta=None, alpha=None):
    if alpha is not None and beta is not None:
        return torch.bmm(batch1, batch2.transpose(1, 2)) * alpha + input * beta
    elif alpha is not None:
        return torch.bmm(batch1, batch2.transpose(1, 2)) * alpha + input
    elif beta is not None:
        return torch.bmm(batch1, batch2.transpose(1, 2)) + input * beta
    else:
        return torch.bmm(batch1, batch2.transpose(1, 2)) + input
