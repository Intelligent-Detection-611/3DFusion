from torch import nn
import os
import nibabel as nib
import torch
from torch.utils.data import Dataset
import numpy as np
import torch.nn.functional as F


class reflect_conv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=4, stride=2, pad=1):
        super(reflect_conv, self).__init__()
        self.conv = nn.Sequential(
            nn.ReflectionPad2d(pad),
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride,
                      padding=0)
        )

    def forward(self, x):
        out = self.conv(x)
        return out

# 新增3D反射填充卷积
class reflect_conv3d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, pad=1):
        super(reflect_conv3d, self).__init__()
        self.conv = nn.Sequential(
            nn.ReflectionPad3d(pad),
            nn.Conv3d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride,
                      padding=0)
        )

    def forward(self, x):
        out = self.conv(x)
        return out

def gradient(input):
    """
    求图像梯度, sobel算子
    :param input:
    :return:
    """

    filter1 = nn.Conv2d(kernel_size=3, in_channels=1, out_channels=1, bias=False, padding=1, stride=1)
    filter2 = nn.Conv2d(kernel_size=3, in_channels=1, out_channels=1, bias=False, padding=1, stride=1)
    filter1.weight.data = torch.tensor([
        [-1., 0., 1.],
        [-2., 0., 2.],
        [-1., 0., 1.]
    ]).reshape(1, 1, 3, 3).cuda()
    filter2.weight.data = torch.tensor([
        [1., 2., 1.],
        [0., 0., 0.],
        [-1., -2., -1.]
    ]).reshape(1, 1, 3, 3).cuda()

    g1 = filter1(input)
    g2 = filter2(input)
    image_gradient = torch.abs(g1) + torch.abs(g2)
    return image_gradient

def clamp(value, min=0., max=1.0):
    """
    将像素值强制约束在[0,1], 以免出现异常斑点
    :param value:
    :param min:
    :param max:
    :return:
    """
    return torch.clamp(value, min=min, max=max)

def RGB2YCrCb(rgb_image):
    """
    将RGB格式转换为YCrCb格式

    :param rgb_image: RGB格式的图像数据
    :return: Y, Cr, Cb
    """

    R = rgb_image[0:1]
    G = rgb_image[1:2]
    B = rgb_image[2:3]
    Y = 0.299 * R + 0.587 * G + 0.114 * B
    Cr = (R - Y) * 0.713 + 0.5
    Cb = (B - Y) * 0.564 + 0.5

    Y = clamp(Y)
    Cr = clamp(Cr)
    Cb = clamp(Cb)
    return Y, Cb, Cr

def YCrCb2RGB(Y, Cb, Cr):
    """
    将YcrCb格式转换为RGB格式

    :param Y:
    :param Cb:
    :param Cr:
    :return:
    """
    ycrcb = torch.cat([Y, Cr, Cb], dim=0)
    C, W, H = ycrcb.shape
    im_flat = ycrcb.reshape(3, -1).transpose(0, 1)
    mat = torch.tensor(
        [[1.0, 1.0, 1.0], [1.403, -0.714, 0.0], [0.0, -0.344, 1.773]]
    ).to(Y.device)
    bias = torch.tensor([0.0 / 255, -0.5, -0.5]).to(Y.device)
    temp = (im_flat + bias).mm(mat)
    out = temp.transpose(0, 1).reshape(C, W, H)
    out = clamp(out)
    return out

def gradient3d(input):
    """
    计算3D体数据的梯度（3D Sobel）。
    input: [B, 1, D, H, W]
    """
    sobel_kernel_x = torch.tensor([
        [[-1, 0, 1], [-3, 0, 3], [-1, 0, 1]],
        [[-3, 0, 3], [-6, 0, 6], [-3, 0, 3]],
        [[-1, 0, 1], [-3, 0, 3], [-1, 0, 1]]
    ], dtype=torch.float32).view(1, 1, 3, 3, 3).to(input.device)

    sobel_kernel_y = torch.tensor([
        [[-1, -3, -1], [0, 0, 0], [1, 3, 1]],
        [[-3, -6, -3], [0, 0, 0], [3, 6, 3]],
        [[-1, -3, -1], [0, 0, 0], [1, 3, 1]]
    ], dtype=torch.float32).view(1, 1, 3, 3, 3).to(input.device)

    sobel_kernel_z = torch.tensor([
        [[-1, -3, -1], [-3, -6, -3], [-1, -3, -1]],
        [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
        [[1, 3, 1], [3, 6, 3], [1, 3, 1]]
    ], dtype=torch.float32).view(1, 1, 3, 3, 3).to(input.device)

    grad_x = F.conv3d(input, sobel_kernel_x, padding=1)
    grad_y = F.conv3d(input, sobel_kernel_y, padding=1)
    grad_z = F.conv3d(input, sobel_kernel_z, padding=1)
    grad = torch.abs(grad_x) + torch.abs(grad_y) + torch.abs(grad_z)
    return grad
