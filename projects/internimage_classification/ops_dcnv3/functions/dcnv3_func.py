# --------------------------------------------------------
# InternImage
# Copyright (c) 2022 OpenGVLab
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------

# Copied from
# https://github.com/OpenGVLab/InternImage/blob/master/classification/models/

from __future__ import absolute_import, division, print_function
import pkg_resources

import DCNv3
import torch
import torch.nn.functional as F
from torch.autograd import Function
from torch.autograd.function import once_differentiable
from torch.cuda.amp import custom_bwd, custom_fwd

dcn_version = float(pkg_resources.get_distribution('DCNv3').version)


class DCNv3Function(Function):

    @staticmethod
    @custom_fwd
    def forward(ctx, input, offset, mask, kernel_h, kernel_w, stride_h,
                stride_w, pad_h, pad_w, dilation_h, dilation_w, group,
                group_channels, offset_scale, im2col_step, remove_center):
        ctx.kernel_h = kernel_h
        ctx.kernel_w = kernel_w
        ctx.stride_h = stride_h
        ctx.stride_w = stride_w
        ctx.pad_h = pad_h
        ctx.pad_w = pad_w
        ctx.dilation_h = dilation_h
        ctx.dilation_w = dilation_w
        ctx.group = group
        ctx.group_channels = group_channels
        ctx.offset_scale = offset_scale
        ctx.im2col_step = im2col_step
        ctx.remove_center = remove_center

        args = [
            input, offset, mask, kernel_h, kernel_w, stride_h, stride_w, pad_h,
            pad_w, dilation_h, dilation_w, group, group_channels, offset_scale,
            ctx.im2col_step
        ]
        if remove_center or dcn_version > 1.0:
            args.append(remove_center)

        output = DCNv3.dcnv3_forward(*args)
        ctx.save_for_backward(input, offset, mask)

        return output

    @staticmethod
    @once_differentiable
    @custom_bwd
    def backward(ctx, grad_output):
        input, offset, mask = ctx.saved_tensors

        args = [
            input, offset, mask, ctx.kernel_h, ctx.kernel_w, ctx.stride_h,
            ctx.stride_w, ctx.pad_h, ctx.pad_w, ctx.dilation_h, ctx.dilation_w,
            ctx.group, ctx.group_channels, ctx.offset_scale,
            grad_output.contiguous(), ctx.im2col_step
        ]
        if ctx.remove_center or dcn_version > 1.0:
            args.append(ctx.remove_center)

        grad_input, grad_offset, grad_mask = \
            DCNv3.dcnv3_backward(*args)

        return grad_input, grad_offset, grad_mask, \
            None, None, None, None, None, None, None,\
            None, None, None, None, None, None

    @staticmethod
    def symbolic(g, input, offset, mask, kernel_h, kernel_w, stride_h,
                 stride_w, pad_h, pad_w, dilation_h, dilation_w, group,
                 group_channels, offset_scale, im2col_step, remove_center):
        """Symbolic function for mmdeploy::DCNv3.

        Returns:
            DCNv3 op for onnx.
        """
        return g.op(
            'mmdeploy::TRTDCNv3',
            input,
            offset,
            mask,
            kernel_h_i=int(kernel_h),
            kernel_w_i=int(kernel_w),
            stride_h_i=int(stride_h),
            stride_w_i=int(stride_w),
            pad_h_i=int(pad_h),
            pad_w_i=int(pad_w),
            dilation_h_i=int(dilation_h),
            dilation_w_i=int(dilation_w),
            group_i=int(group),
            group_channels_i=int(group_channels),
            offset_scale_f=float(offset_scale),
            im2col_step_i=int(im2col_step),
            remove_center=int(remove_center),
        )


def _get_reference_points(spatial_shapes,
                          device,
                          kernel_h,
                          kernel_w,
                          dilation_h,
                          dilation_w,
                          pad_h=0,
                          pad_w=0,
                          stride_h=1,
                          stride_w=1):
    _, H_, W_, _ = spatial_shapes
    H_out = (H_ - (dilation_h * (kernel_h - 1) + 1)) // stride_h + 1
    W_out = (W_ - (dilation_w * (kernel_w - 1) + 1)) // stride_w + 1

    ref_y, ref_x = torch.meshgrid(
        torch.linspace(
            # pad_h + 0.5,
            # H_ - pad_h - 0.5,
            (dilation_h * (kernel_h - 1)) // 2 + 0.5,
            (dilation_h * (kernel_h - 1)) // 2 + 0.5 + (H_out - 1) * stride_h,
            H_out,
            dtype=torch.float32,
            device=device),
        torch.linspace(
            # pad_w + 0.5,
            # W_ - pad_w - 0.5,
            (dilation_w * (kernel_w - 1)) // 2 + 0.5,
            (dilation_w * (kernel_w - 1)) // 2 + 0.5 + (W_out - 1) * stride_w,
            W_out,
            dtype=torch.float32,
            device=device))
    ref_y = ref_y.reshape(-1)[None] / H_
    ref_x = ref_x.reshape(-1)[None] / W_

    ref = torch.stack((ref_x, ref_y), -1).reshape(1, H_out, W_out, 1, 2)

    return ref


def _generate_dilation_grids(spatial_shapes, kernel_h, kernel_w, dilation_h,
                             dilation_w, group, device):
    _, H_, W_, _ = spatial_shapes
    points_list = []
    x, y = torch.meshgrid(
        torch.linspace(
            -((dilation_w * (kernel_w - 1)) // 2),
            -((dilation_w * (kernel_w - 1)) // 2) +
            (kernel_w - 1) * dilation_w,
            kernel_w,
            dtype=torch.float32,
            device=device),
        torch.linspace(
            -((dilation_h * (kernel_h - 1)) // 2),
            -((dilation_h * (kernel_h - 1)) // 2) +
            (kernel_h - 1) * dilation_h,
            kernel_h,
            dtype=torch.float32,
            device=device))

    points_list.extend([x / W_, y / H_])
    grid = torch.stack(points_list, -1).reshape(-1, 1, 2).\
        repeat(1, group, 1).permute(1, 0, 2)
    grid = grid.reshape(1, 1, 1, group * kernel_h * kernel_w, 2)

    return grid


def remove_center_sampling_locations(sampling_locations, kernel_w, kernel_h):
    idx = list(range(sampling_locations.shape[-2]))
    C = (kernel_w * kernel_h - 1) // 2
    idx = [i for i in idx if i != C and (i - C) % (C * 2 + 1) != 0]
    sampling_locations = sampling_locations[:, :, :, idx, :]
    return sampling_locations


def dcnv3_core_pytorch(input, offset, mask, kernel_h, kernel_w, stride_h,
                       stride_w, pad_h, pad_w, dilation_h, dilation_w, group,
                       group_channels, offset_scale, remove_center):
    # for debug and test only,
    # need to use cuda version instead

    if remove_center and (kernel_h % 2 == 0 or kernel_w % 2 == 0
                          or kernel_w != kernel_h):
        raise ValueError(
            'remove_center is only compatible with square odd kernel size.')

    input = F.pad(input, [0, 0, pad_h, pad_h, pad_w, pad_w])
    N_, H_in, W_in, _ = input.shape
    _, H_out, W_out, _ = offset.shape

    ref = _get_reference_points(input.shape, input.device, kernel_h, kernel_w,
                                dilation_h, dilation_w, pad_h, pad_w, stride_h,
                                stride_w)
    grid = _generate_dilation_grids(input.shape, kernel_h, kernel_w,
                                    dilation_h, dilation_w, group,
                                    input.device)
    spatial_norm = torch.tensor([W_in, H_in]).reshape(1, 1, 1, 2).\
        repeat(1, 1, 1, group*(kernel_h*kernel_w-remove_center)).\
        to(input.device)

    sampling_locations = (ref + grid * offset_scale).repeat(N_, 1, 1, 1, 1)
    if remove_center:
        sampling_locations = remove_center_sampling_locations(
            sampling_locations, kernel_w=kernel_w, kernel_h=kernel_h)
    sampling_locations = sampling_locations.flatten(3, 4)
    sampling_locations = sampling_locations + \
        offset * offset_scale / spatial_norm

    P_ = kernel_h * kernel_w - remove_center
    sampling_grids = 2 * sampling_locations - 1
    # N_, H_in, W_in, group*group_channels ->
    # N_, H_in*W_in, group*group_channels ->
    # N_, group*group_channels, H_in*W_in ->
    # N_*group, group_channels, H_in, W_in
    input_ = input.view(N_, H_in*W_in, group*group_channels).transpose(1, 2).\
        reshape(N_*group, group_channels, H_in, W_in)
    # N_, H_out, W_out, group*P_*2 ->
    # N_, H_out*W_out, group, P_, 2 ->
    # N_, group, H_out*W_out, P_, 2 ->
    # N_*group, H_out*W_out, P_, 2
    sampling_grid_ = sampling_grids.view(N_, H_out*W_out, group, P_, 2).\
        transpose(1, 2).flatten(0, 1)
    # N_*group, group_channels, H_out*W_out, P_
    sampling_input_ = F.grid_sample(
        input_,
        sampling_grid_,
        mode='bilinear',
        padding_mode='zeros',
        align_corners=False)

    # (N_, H_out, W_out, group*P_) ->
    # N_, H_out*W_out, group, P_ ->
    # (N_, group, H_out*W_out, P_) ->
    # (N_*group, 1, H_out*W_out, P_)
    mask = mask.view(N_, H_out*W_out, group, P_).transpose(1, 2).\
        reshape(N_*group, 1, H_out*W_out, P_)
    output = (sampling_input_ * mask).sum(-1).view(N_, group * group_channels,
                                                   H_out * W_out)

    return output.transpose(1, 2).reshape(N_, H_out, W_out, -1).contiguous()
