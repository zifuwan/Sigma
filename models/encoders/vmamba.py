import os
import time
import math
import copy
from functools import partial
from typing import Optional, Callable, Any
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from einops import rearrange, repeat
from timm.models.layers import DropPath, trunc_normal_
from fvcore.nn import FlopCountAnalysis, flop_count_str, flop_count, parameter_count
DropPath.__repr__ = lambda self: f"timm.DropPath({self.drop_prob})"

# import mamba_ssm.selective_scan_fn (in which causal_conv1d is needed)
try:
    from mamba_ssm.ops.selective_scan_interface import selective_scan_fn, selective_scan_ref
except:
    pass

# an alternative for mamba_ssm
try:
    from selective_scan import selective_scan_fn as selective_scan_fn_v1
    from selective_scan import selective_scan_ref as selective_scan_ref_v1
except:
    pass

# cross selective scan ===============================
if True:
    import selective_scan_cuda_core as selective_scan_cuda
    
    class SelectiveScan(torch.autograd.Function):
        @staticmethod
        @torch.cuda.amp.custom_fwd(cast_inputs=torch.float32)
        def forward(ctx, u, delta, A, B, C, D=None, delta_bias=None, delta_softplus=False, nrows=1):
            assert nrows in [1, 2, 3, 4], f"{nrows}" # 8+ is too slow to compile
            assert u.shape[1] % (B.shape[1] * nrows) == 0, f"{nrows}, {u.shape}, {B.shape}"
            ctx.delta_softplus = delta_softplus
            ctx.nrows = nrows

            # all in float
            if u.stride(-1) != 1:
                u = u.contiguous()
            if delta.stride(-1) != 1:
                delta = delta.contiguous()
            if D is not None:
                D = D.contiguous()
            if B.stride(-1) != 1:
                B = B.contiguous()
            if C.stride(-1) != 1:
                C = C.contiguous()
            if B.dim() == 3:
                B = B.unsqueeze(dim=1)
                ctx.squeeze_B = True
            if C.dim() == 3:
                C = C.unsqueeze(dim=1)
                ctx.squeeze_C = True

            out, x, *rest = selective_scan_cuda.fwd(u, delta, A, B, C, D, delta_bias, delta_softplus, nrows)
            
            ctx.save_for_backward(u, delta, A, B, C, D, delta_bias, x)
            return out

        @staticmethod
        @torch.cuda.amp.custom_bwd
        def backward(ctx, dout, *args):
            u, delta, A, B, C, D, delta_bias, x = ctx.saved_tensors
            if dout.stride(-1) != 1:
                dout = dout.contiguous()
            du, ddelta, dA, dB, dC, dD, ddelta_bias, *rest = selective_scan_cuda.bwd(
                u, delta, A, B, C, D, delta_bias, dout, x, ctx.delta_softplus, 1
                # u, delta, A, B, C, D, delta_bias, dout, x, ctx.delta_softplus, ctx.nrows,
            )
            dB = dB.squeeze(1) if getattr(ctx, "squeeze_B", False) else dB
            dC = dC.squeeze(1) if getattr(ctx, "squeeze_C", False) else dC
            return (du, ddelta, dA, dB, dC, dD, ddelta_bias, None, None)

    class CrossScan(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x: torch.Tensor):
            B, C, H, W = x.shape
            ctx.shape = (B, C, H, W)
            xs = x.new_empty((B, 4, C, H * W))
            xs[:, 0] = x.flatten(2, 3)
            xs[:, 1] = x.transpose(dim0=2, dim1=3).flatten(2, 3)
            xs[:, 2:4] = torch.flip(xs[:, 0:2], dims=[-1])
            return xs
        
        @staticmethod
        def backward(ctx, ys: torch.Tensor):
            # out: (b, k, d, l)
            B, C, H, W = ctx.shape
            L = H * W
            ys = ys[:, 0:2] + ys[:, 2:4].flip(dims=[-1]).view(B, 2, -1, L)
            y = ys[:, 0] + ys[:, 1].view(B, -1, W, H).transpose(dim0=2, dim1=3).contiguous().view(B, -1, L)
            return y.view(B, -1, H, W)
    
    class CrossMerge(torch.autograd.Function):
        @staticmethod
        def forward(ctx, ys: torch.Tensor):
            B, K, D, H, W = ys.shape
            ctx.shape = (H, W)
            ys = ys.view(B, K, D, -1)
            ys = ys[:, 0:2] + ys[:, 2:4].flip(dims=[-1]).view(B, 2, D, -1)
            y = ys[:, 0] + ys[:, 1].view(B, -1, W, H).transpose(dim0=2, dim1=3).contiguous().view(B, D, -1)
            return y
        
        @staticmethod
        def backward(ctx, x: torch.Tensor):
            # B, D, L = x.shape
            # out: (b, k, d, l)
            H, W = ctx.shape
            B, C, L = x.shape
            xs = x.new_empty((B, 4, C, L))
            xs[:, 0] = x
            xs[:, 1] = x.view(B, C, H, W).transpose(dim0=2, dim1=3).flatten(2, 3)
            xs[:, 2:4] = torch.flip(xs[:, 0:2], dims=[-1])
            xs = xs.view(B, 4, C, H, W)
            return xs, None, None
        
    class CrossScan_multimodal(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x_rgb: torch.Tensor, x_e: torch.Tensor):
            # B, C, H, W -> B, 2, C, 2 * H * W
            B, C, H, W = x_rgb.shape
            ctx.shape = (B, C, H, W)
            xs_fuse = x_rgb.new_empty((B, 2, C, 2 * H * W))
            xs_fuse[:, 0] = torch.concat([x_rgb.flatten(2, 3), x_e.flatten(2, 3)], dim=2)
            xs_fuse[:, 1] = torch.flip(xs_fuse[:, 0], dims=[-1])
            return xs_fuse
        
        @staticmethod
        def backward(ctx, ys: torch.Tensor):
            # out: (b, 2, d, l)
            B, C, H, W = ctx.shape
            L = 2 * H * W
            ys = ys[:, 0] + ys[:, 1].flip(dims=[-1]).view(B, 1, -1, L) # B, 1, d, 2 * H * W
            ys = ys[:, 0] + ys[:, 1] # B, d, 2 * H * W
            # get B, d, H*W
            return ys[:, :, 0:H*W].view(B, -1, H, W), ys[:, :, H*W:2*H*W].view(B, -1, H, W) 
         
    class CrossMerge_multimodal(torch.autograd.Function):
        @staticmethod
        def forward(ctx, ys: torch.Tensor):
            B, K, D, L = ys.shape
            # ctx.shape = (H, W)
            # ys = ys.view(B, K, D, -1)
            ys = ys[:, 0] + ys[:, 1].flip(dims=[-1]) # B, d, 2 * H * W, broadcast
            # y = ys[:, :, 0:L//2] + ys[:, :, L//2:L]
            return ys[:, :, 0:L//2], ys[:, :, L//2:L]
        
        @staticmethod
        def backward(ctx, x1: torch.Tensor, x2: torch.Tensor):
            # B, D, L = x.shape
            # out: (b, k, d, l)
            # H, W = ctx.shape
            B, C, L = x1.shape
            xs = x1.new_empty((B, 2, C, 2*L))
            xs[:, 0] = torch.cat([x1, x2], dim=2)
            xs[:, 1] = torch.flip(xs[:, 0], dims=[-1])
            xs = xs.view(B, 2, C, 2*L)
            return xs, None, None

    def cross_selective_scan(
        x: torch.Tensor=None, 
        x_proj_weight: torch.Tensor=None,
        x_proj_bias: torch.Tensor=None,
        dt_projs_weight: torch.Tensor=None,
        dt_projs_bias: torch.Tensor=None,
        A_logs: torch.Tensor=None,
        Ds: torch.Tensor=None,
        out_norm: torch.nn.Module=None,
        softmax_version=False,
        nrows = -1,
        delta_softplus = True,
    ):
        B, D, H, W = x.shape
        D, N = A_logs.shape
        K, D, R = dt_projs_weight.shape
        L = H * W

        if nrows < 1:
            if D % 4 == 0:
                nrows = 4
            elif D % 3 == 0:
                nrows = 3
            elif D % 2 == 0:
                nrows = 2
            else:
                nrows = 1

        xs = CrossScan.apply(x)
        
        x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs, x_proj_weight)
        if x_proj_bias is not None:
            x_dbl = x_dbl + x_proj_bias.view(1, K, -1, 1)
        dts, Bs, Cs = torch.split(x_dbl, [R, N, N], dim=2)
        dts = torch.einsum("b k r l, k d r -> b k d l", dts, dt_projs_weight)

        xs = xs.view(B, -1, L).to(torch.float)
        dts = dts.contiguous().view(B, -1, L).to(torch.float)
        As = -torch.exp(A_logs.to(torch.float)) # (k * c, d_state)
        Bs = Bs.contiguous().to(torch.float)
        Cs = Cs.contiguous().to(torch.float)
        Ds = Ds.to(torch.float) # (K * c)
        delta_bias = dt_projs_bias.view(-1).to(torch.float)
         
        # to enable fvcore.nn.jit_analysis: inputs[i].debugName
        def selective_scan(u, delta, A, B, C, D=None, delta_bias=None, delta_softplus=True, nrows=1):
            return SelectiveScan.apply(u, delta, A, B, C, D, delta_bias, delta_softplus, nrows)

        ys: torch.Tensor = selective_scan(
            xs, dts, As, Bs, Cs, Ds, delta_bias, delta_softplus, nrows,
        ).view(B, K, -1, H, W)
        
        y = CrossMerge.apply(ys)

        if softmax_version:
            y = y.softmax(y, dim=-1).to(x.dtype)
            y = y.transpose(dim0=1, dim1=2).contiguous().view(B, H, W, -1)
        else:
            y = y.transpose(dim0=1, dim1=2).contiguous().view(B, H, W, -1)
            y = out_norm(y).to(x.dtype)
        
        return y
    
    
    def selective_scan_1d(
        x: torch.Tensor=None, 
        x_proj_weight: torch.Tensor=None,
        x_proj_bias: torch.Tensor=None,
        dt_projs_weight: torch.Tensor=None,
        dt_projs_bias: torch.Tensor=None,
        A_logs: torch.Tensor=None,
        Ds: torch.Tensor=None,
        out_norm: torch.nn.Module=None,
        softmax_version=False,
        nrows = -1,
        delta_softplus = True,
    ):
        A_logs = A_logs[: A_logs.shape[0] // 4]
        Ds = Ds[: Ds.shape[0] // 4]
        B, D, H, W = x.shape
        D, N = A_logs.shape
        # get 1st of dt_projs_weight
        x_proj_weight = x_proj_weight[0].unsqueeze(0)
        x_proj_bias = x_proj_bias[0].unsqueeze(0) if x_proj_bias is not None else None
        dt_projs_weight = dt_projs_weight[0].unsqueeze(0)
        dt_projs_bias = dt_projs_bias[0].unsqueeze(0) if dt_projs_bias is not None else None
        K, D, R = dt_projs_weight.shape # K=1
        L = H * W

        if nrows < 1:
            if D % 4 == 0:
                nrows = 4
            elif D % 3 == 0:
                nrows = 3
            elif D % 2 == 0:
                nrows = 2
            else:
                nrows = 1

        # xs = CrossScan.apply(x)
        xs = x.view(B, -1, L).unsqueeze(dim=1)
        
        x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs, x_proj_weight)
        if x_proj_bias is not None:
            x_dbl = x_dbl + x_proj_bias.view(1, K, -1, 1)
        dts, Bs, Cs = torch.split(x_dbl, [R, N, N], dim=2)
        dts = torch.einsum("b k r l, k d r -> b k d l", dts, dt_projs_weight)

        xs = xs.view(B, -1, L).to(torch.float)
        dts = dts.contiguous().view(B, -1, L).to(torch.float)
        As = -torch.exp(A_logs.to(torch.float)) # (k * c, d_state)
        Bs = Bs.contiguous().to(torch.float)
        Cs = Cs.contiguous().to(torch.float)
        Ds = Ds.to(torch.float) # (K * c)
        delta_bias = dt_projs_bias.view(-1).to(torch.float)
         
        # to enable fvcore.nn.jit_analysis: inputs[i].debugName
        def selective_scan(u, delta, A, B, C, D=None, delta_bias=None, delta_softplus=True, nrows=1):
            return SelectiveScan.apply(u, delta, A, B, C, D, delta_bias, delta_softplus, nrows)

        ys: torch.Tensor = selective_scan(
            xs, dts, As, Bs, Cs, Ds, delta_bias, delta_softplus, nrows,
        ).view(B, K, -1, L)
        
        # y = CrossMerge.apply(ys)

        if softmax_version:
            y = y.softmax(y, dim=-1).to(x.dtype)
            y = ys[:, 0].transpose(dim0=1, dim1=2).contiguous().view(B, H, W, -1)
        else:
            y = ys[:, 0].transpose(dim0=1, dim1=2).contiguous().view(B, H, W, -1)
            y = out_norm(y).to(x.dtype)
        
        return y
    
    
    def cross_selective_scan_multimodal_k1(
        x_rgb: torch.Tensor=None, 
        x_e: torch.Tensor=None,
        x_proj_weight: torch.Tensor=None,
        x_proj_bias: torch.Tensor=None,
        dt_projs_weight: torch.Tensor=None,
        dt_projs_bias: torch.Tensor=None,
        A_logs: torch.Tensor=None,
        Ds: torch.Tensor=None,
        out_norm: torch.nn.Module=None,
        softmax_version=False,
        nrows = -1,
        delta_softplus = True,
    ):
        B, D, H, W = x_rgb.shape
        D, N = A_logs.shape
        K, D, R = dt_projs_weight.shape
        L = 2 * H * W

        if nrows < 1:
            if D % 4 == 0:
                nrows = 4
            elif D % 3 == 0:
                nrows = 3
            elif D % 2 == 0:
                nrows = 2
            else:
                nrows = 1

        # x_fuse = CrossScan_multimodal.apply(x_rgb, x_e) # B, C, H, W -> B, 1, C, 2 * H * W
        B, C, H, W = x_rgb.shape
        x_fuse = x_rgb.new_empty((B, 1, C, 2 * H * W))
        x_fuse[:, 0] = torch.concat([x_rgb.flatten(2, 3), x_e.flatten(2, 3)], dim=2)
        
        x_dbl = torch.einsum("b k d l, k c d -> b k c l", x_fuse, x_proj_weight)
        if x_proj_bias is not None:
            x_dbl = x_dbl + x_proj_bias.view(1, K, -1, 1)
        dts, Bs, Cs = torch.split(x_dbl, [R, N, N], dim=2)
        dts = torch.einsum("b k r l, k d r -> b k d l", dts, dt_projs_weight)

        x_fuse = x_fuse.view(B, -1, L).to(torch.float)
        dts = dts.contiguous().view(B, -1, L).to(torch.float)
        As = -torch.exp(A_logs.to(torch.float)) # (k * c, d_state)
        Bs = Bs.contiguous().to(torch.float)
        Cs = Cs.contiguous().to(torch.float)
        Ds = Ds.to(torch.float) # (K * c)
        delta_bias = dt_projs_bias.view(-1).to(torch.float)
         
        # to enable fvcore.nn.jit_analysis: inputs[i].debugName
        def selective_scan(u, delta, A, B, C, D=None, delta_bias=None, delta_softplus=True, nrows=1):
            return SelectiveScan.apply(u, delta, A, B, C, D, delta_bias, delta_softplus, nrows)

        ys: torch.Tensor = selective_scan(
            x_fuse, dts, As, Bs, Cs, Ds, delta_bias, delta_softplus, nrows,
        ).view(B, K, -1, 2*H*W)
        
        # y = CrossMerge_multimodal.apply(ys)
        y = ys[:, 0, :, 0:L//2] + ys[:, 0, :, L//2:L]

        if softmax_version:
            y = y.softmax(y, dim=-1).to(x_rgb.dtype)
            y = y.transpose(dim0=1, dim1=2).contiguous().view(B, H, W, -1)
        else:
            y = y.transpose(dim0=1, dim1=2).contiguous().view(B, H, W, -1)
            y = out_norm(y).to(x_rgb.dtype)
        
        return y

    def cross_selective_scan_multimodal_k2(
        x_rgb: torch.Tensor=None, 
        x_e: torch.Tensor=None,
        x_proj_weight: torch.Tensor=None,
        x_proj_bias: torch.Tensor=None,
        dt_projs_weight: torch.Tensor=None,
        dt_projs_bias: torch.Tensor=None,
        A_logs: torch.Tensor=None,
        Ds: torch.Tensor=None,
        out_norm1: torch.nn.Module=None,
        out_norm2: torch.nn.Module=None,
        softmax_version=False,
        nrows = -1,
        delta_softplus = True,
    ):
        B, D, H, W = x_rgb.shape
        D, N = A_logs.shape
        K, D, R = dt_projs_weight.shape
        L = 2 * H * W

        if nrows < 1:
            if D % 4 == 0:
                nrows = 4
            elif D % 3 == 0:
                nrows = 3
            elif D % 2 == 0:
                nrows = 2
            else:
                nrows = 1

        x_fuse = CrossScan_multimodal.apply(x_rgb, x_e) # B, C, H, W -> B, 2, C, 2 * H * W
       
        x_dbl = torch.einsum("b k d l, k c d -> b k c l", x_fuse, x_proj_weight)
        if x_proj_bias is not None:
            x_dbl = x_dbl + x_proj_bias.view(1, K, -1, 1)
        dts, Bs, Cs = torch.split(x_dbl, [R, N, N], dim=2)
        dts = torch.einsum("b k r l, k d r -> b k d l", dts, dt_projs_weight)

        x_fuse = x_fuse.view(B, -1, L).to(torch.float)
        dts = dts.contiguous().view(B, -1, L).to(torch.float)
        As = -torch.exp(A_logs.to(torch.float)) # (k * c, d_state)
        Bs = Bs.contiguous().to(torch.float)
        Cs = Cs.contiguous().to(torch.float)
        Ds = Ds.to(torch.float) # (K * c)
        delta_bias = dt_projs_bias.view(-1).to(torch.float)
         
        # to enable fvcore.nn.jit_analysis: inputs[i].debugName
        def selective_scan(u, delta, A, B, C, D=None, delta_bias=None, delta_softplus=True, nrows=1):
            return SelectiveScan.apply(u, delta, A, B, C, D, delta_bias, delta_softplus, nrows)

        ys: torch.Tensor = selective_scan(
            x_fuse, dts, As, Bs, Cs, Ds, delta_bias, delta_softplus, nrows,
        ).view(B, K, -1, 2*H*W)
        
        y_rgb, y_e = CrossMerge_multimodal.apply(ys)

        y_rgb = y_rgb.transpose(dim0=1, dim1=2).contiguous().view(B, H, W, -1)
        y_e = y_e.transpose(dim0=1, dim1=2).contiguous().view(B, H, W, -1)
        y_rgb = out_norm1(y_rgb).to(x_rgb.dtype)
        y_e = out_norm2(y_e).to(x_e.dtype)
        
        return y_rgb, y_e


# fvcore flops =======================================

def flops_selective_scan_fn(B=1, L=256, D=768, N=16, with_D=True, with_Z=False, with_Group=True, with_complex=False):
    """
    u: r(B D L)
    delta: r(B D L)
    A: r(D N)
    B: r(B N L)
    C: r(B N L)
    D: r(D)
    z: r(B D L)
    delta_bias: r(D), fp32
    
    ignores:
        [.float(), +, .softplus, .shape, new_zeros, repeat, stack, to(dtype), silu] 
    """
    assert not with_complex 
    # https://github.com/state-spaces/mamba/issues/110
    flops = 9 * B * L * D * N
    if with_D:
        flops += B * D * L
    if with_Z:
        flops += B * D * L    
    return flops


def flops_selective_scan_ref(B=1, L=256, D=768, N=16, with_D=True, with_Z=False, with_Group=True, with_complex=False):
    """
    u: r(B D L)
    delta: r(B D L)
    A: r(D N)
    B: r(B N L)
    C: r(B N L)
    D: r(D)
    z: r(B D L)
    delta_bias: r(D), fp32
    
    ignores:
        [.float(), +, .softplus, .shape, new_zeros, repeat, stack, to(dtype), silu] 
    """
    import numpy as np
    
    # fvcore.nn.jit_handles
    def get_flops_einsum(input_shapes, equation):
        np_arrs = [np.zeros(s) for s in input_shapes]
        optim = np.einsum_path(equation, *np_arrs, optimize="optimal")[1]
        for line in optim.split("\n"):
            if "optimized flop" in line.lower():
                # divided by 2 because we count MAC (multiply-add counted as one flop)
                flop = float(np.floor(float(line.split(":")[-1]) / 2))
                return flop
    

    assert not with_complex

    flops = 0 # below code flops = 0
    if False:
        ...
        """
        dtype_in = u.dtype
        u = u.float()
        delta = delta.float()
        if delta_bias is not None:
            delta = delta + delta_bias[..., None].float()
        if delta_softplus:
            delta = F.softplus(delta)
        batch, dim, dstate = u.shape[0], A.shape[0], A.shape[1]
        is_variable_B = B.dim() >= 3
        is_variable_C = C.dim() >= 3
        if A.is_complex():
            if is_variable_B:
                B = torch.view_as_complex(rearrange(B.float(), "... (L two) -> ... L two", two=2))
            if is_variable_C:
                C = torch.view_as_complex(rearrange(C.float(), "... (L two) -> ... L two", two=2))
        else:
            B = B.float()
            C = C.float()
        x = A.new_zeros((batch, dim, dstate))
        ys = []
        """

    flops += get_flops_einsum([[B, D, L], [D, N]], "bdl,dn->bdln")
    if with_Group:
        flops += get_flops_einsum([[B, D, L], [B, N, L], [B, D, L]], "bdl,bnl,bdl->bdln")
    else:
        flops += get_flops_einsum([[B, D, L], [B, D, N, L], [B, D, L]], "bdl,bdnl,bdl->bdln")
    if False:
        ...
        """
        deltaA = torch.exp(torch.einsum('bdl,dn->bdln', delta, A))
        if not is_variable_B:
            deltaB_u = torch.einsum('bdl,dn,bdl->bdln', delta, B, u)
        else:
            if B.dim() == 3:
                deltaB_u = torch.einsum('bdl,bnl,bdl->bdln', delta, B, u)
            else:
                B = repeat(B, "B G N L -> B (G H) N L", H=dim // B.shape[1])
                deltaB_u = torch.einsum('bdl,bdnl,bdl->bdln', delta, B, u)
        if is_variable_C and C.dim() == 4:
            C = repeat(C, "B G N L -> B (G H) N L", H=dim // C.shape[1])
        last_state = None
        """
    
    in_for_flops = B * D * N   
    if with_Group:
        in_for_flops += get_flops_einsum([[B, D, N], [B, D, N]], "bdn,bdn->bd")
    else:
        in_for_flops += get_flops_einsum([[B, D, N], [B, N]], "bdn,bn->bd")
    flops += L * in_for_flops 
    if False:
        ...
        """
        for i in range(u.shape[2]):
            x = deltaA[:, :, i] * x + deltaB_u[:, :, i]
            if not is_variable_C:
                y = torch.einsum('bdn,dn->bd', x, C)
            else:
                if C.dim() == 3:
                    y = torch.einsum('bdn,bn->bd', x, C[:, :, i])
                else:
                    y = torch.einsum('bdn,bdn->bd', x, C[:, :, :, i])
            if i == u.shape[2] - 1:
                last_state = x
            if y.is_complex():
                y = y.real * 2
            ys.append(y)
        y = torch.stack(ys, dim=2) # (batch dim L)
        """

    if with_D:
        flops += B * D * L
    if with_Z:
        flops += B * D * L
    if False:
        ...
        """
        out = y if D is None else y + u * rearrange(D, "d -> d 1")
        if z is not None:
            out = out * F.silu(z)
        out = out.to(dtype=dtype_in)
        """
    
    return flops


def print_jit_input_names(inputs):
    # tensor.11, dt.1, A.1, B.1, C.1, D.1, z.1, None
    try: 
        print("input params: ", end=" ", flush=True)
        for i in range(10):
            print(inputs[i].debugName(), end=" ", flush=True)
    except Exception as e:
        pass
    print("", flush=True)


def selective_scan_flop_jit(inputs, outputs):
    print_jit_input_names(inputs)

    # xs, dts, As, Bs, Cs, Ds (skip), z (skip), dt_projs_bias (skip)
    assert inputs[0].debugName().startswith("xs") # (B, D, L)
    assert inputs[1].debugName().startswith("dts") # (B, D, L)
    assert inputs[2].debugName().startswith("As") # (D, N)
    assert inputs[3].debugName().startswith("Bs") # (D, N)
    assert inputs[4].debugName().startswith("Cs") # (D, N)
    with_Group = len(inputs[3].type().sizes()) == 4
    with_D = inputs[5].debugName().startswith("Ds")
    if not with_D:
        with_z = len(inputs) > 5 and inputs[5].debugName().startswith("z")
    else:
        with_z = len(inputs) > 6 and inputs[6].debugName().startswith("z")
    B, D, L = inputs[0].type().sizes()
    N = inputs[2].type().sizes()[1]
    flops = flops_selective_scan_fn(B=B, L=L, D=D, N=N, with_D=with_D, with_Z=with_z, with_Group=with_Group)
    # flops = flops_selective_scan_ref(B=B, L=L, D=D, N=N, with_D=with_D, with_Z=with_z, with_Group=with_Group)
    return flops

# =====================================================

class PatchMerging2D(nn.Module):
    def __init__(self, dim, out_dim=-1, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, (2 * dim) if out_dim < 0 else out_dim, bias=False)
        self.norm = norm_layer(4 * dim)

    @staticmethod
    def _patch_merging_pad(x: torch.Tensor):
        H, W, _ = x.shape[-3:]
        if (W % 2 != 0) or (H % 2 != 0):
            x = F.pad(x, (0, 0, 0, W % 2, 0, H % 2))
        x0 = x[..., 0::2, 0::2, :]  # ... H/2 W/2 C
        x1 = x[..., 1::2, 0::2, :]  # ... H/2 W/2 C
        x2 = x[..., 0::2, 1::2, :]  # ... H/2 W/2 C
        x3 = x[..., 1::2, 1::2, :]  # ... H/2 W/2 C
        x = torch.cat([x0, x1, x2, x3], -1)  # ... H/2 W/2 4*C
        return x

    def forward(self, x):
        x = self._patch_merging_pad(x)
        x = self.norm(x)
        x = self.reduction(x)

        return x


DEV = False
class SS2D(nn.Module):
    def __init__(
        self,
        # basic dims ===========
        d_model=96,
        d_state=16,
        ssm_ratio=2,
        dt_rank="auto",
        # dwconv ===============
        # d_conv=-1, # < 2 means no conv 
        d_conv=3, # < 2 means no conv 
        conv_bias=True,
        # ======================
        dropout=0.,
        bias=False,
        # dt init ==============
        dt_min=0.001,
        dt_max=0.1,
        dt_init="random",
        dt_scale=1.0,
        dt_init_floor=1e-4,
        # ======================
        softmax_version=False,
        # ======================
        **kwargs,
    ):
        if DEV:
            d_conv = -1
            
        factory_kwargs = {"device": None, "dtype": None}
        super().__init__()
        self.softmax_version = softmax_version
        self.d_model = d_model
        self.d_state = math.ceil(self.d_model / 6) if d_state == "auto" else d_state # 20240109
        self.d_conv = d_conv
        self.expand = ssm_ratio
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank

        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs)
        
        # conv =======================================
        if self.d_conv > 1:
            self.conv2d = nn.Conv2d(
                in_channels=self.d_inner,
                out_channels=self.d_inner,
                groups=self.d_inner,
                bias=conv_bias,
                kernel_size=d_conv,
                padding=(d_conv - 1) // 2,
                **factory_kwargs,
            )
            self.act = nn.SiLU()

        # x proj; dt proj ============================
        self.K = 4 if not (self.forward_core == self.forward_corev1_share_ssm) else 1
        # VMamaba set K=4, while original SSM set K=1
        if self.forward_core == self.forward_corev0:
            self.K = 1
        self.x_proj = [
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs)
            for _ in range(self.K)
        ]
        self.x_proj_weight = nn.Parameter(torch.stack([t.weight for t in self.x_proj], dim=0)) # (K, N, inner)
        del self.x_proj

        self.dt_projs = [
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs)
            for _ in range(self.K)
        ]
        self.dt_projs_weight = nn.Parameter(torch.stack([t.weight for t in self.dt_projs], dim=0)) # (K, inner, rank)
        self.dt_projs_bias = nn.Parameter(torch.stack([t.bias for t in self.dt_projs], dim=0)) # (K, inner)
        del self.dt_projs
        
        # A, D =======================================
        self.K2 = self.K if not (self.forward_core == self.forward_corev1_share_a) else 1
        # VMamaba set K=4, while original SSM set K=1
        if self.forward_core == self.forward_corev0:
            self.K2 = 1
        self.A_logs = self.A_log_init(self.d_state, self.d_inner, copies=self.K2, merge=True) # (K * D, N)
        self.Ds = self.D_init(self.d_inner, copies=self.K2, merge=True) # (K * D)

        # out proj =======================================
        if not self.softmax_version:
            self.out_norm = nn.LayerNorm(self.d_inner)
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)
        self.dropout = nn.Dropout(dropout) if dropout > 0. else nn.Identity()

    @staticmethod
    def dt_init(dt_rank, d_inner, dt_scale=1.0, dt_init="random", dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4, **factory_kwargs):
        dt_proj = nn.Linear(dt_rank, d_inner, bias=True, **factory_kwargs)

        # Initialize special dt projection to preserve variance at initialization
        dt_init_std = dt_rank**-0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        # Initialize dt bias so that F.softplus(dt_bias) is between dt_min and dt_max
        dt = torch.exp(
            torch.rand(d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            dt_proj.bias.copy_(inv_dt)
        # Our initialization would set all Linear.bias to zero, need to mark this one as _no_reinit
        # dt_proj.bias._no_reinit = True
        
        return dt_proj

    @staticmethod
    def A_log_init(d_state, d_inner, copies=-1, device=None, merge=True):
        # S4D real initialization
        A = repeat(
            torch.arange(1, d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=d_inner,
        ).contiguous()
        A_log = torch.log(A)  # Keep A_log in fp32
        if copies > 0:
            A_log = repeat(A_log, "d n -> r d n", r=copies)
            if merge:
                A_log = A_log.flatten(0, 1)
        A_log = nn.Parameter(A_log)
        A_log._no_weight_decay = True
        return A_log

    @staticmethod
    def D_init(d_inner, copies=-1, device=None, merge=True):
        # D "skip" parameter
        D = torch.ones(d_inner, device=device)
        if copies > 0:
            D = repeat(D, "n1 -> r n1", r=copies)
            if merge:
                D = D.flatten(0, 1)
        D = nn.Parameter(D)  # Keep in fp32
        D._no_weight_decay = True
        return D

    def forward_corev0(self, x: torch.Tensor):
        selective_scan = selective_scan_fn_v1

        B, C, H, W = x.shape
        L = H * W
        K = 1 #4

        # x_hwwh = torch.stack([x.view(B, -1, L), torch.transpose(x, dim0=2, dim1=3).contiguous().view(B, -1, L)], dim=1).view(B, 2, -1, L)
        # xs = torch.cat([x_hwwh, torch.flip(x_hwwh, dims=[-1])], dim=1) # (b, k, d, l)
        xs = x.view(B, -1, L).unsqueeze(dim=1)

        x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs, self.x_proj_weight)
        # x_dbl = x_dbl + self.x_proj_bias.view(1, K, -1, 1)
        dts, Bs, Cs = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=2)
        dts = torch.einsum("b k r l, k d r -> b k d l", dts, self.dt_projs_weight)

        xs = xs.float().view(B, -1, L) # (b, k * d, l)
        dts = dts.contiguous().float().view(B, -1, L) # (b, k * d, l)
        Bs = Bs.float() # (b, k, d_state, l)
        Cs = Cs.float() # (b, k, d_state, l)
        
        As = -torch.exp(self.A_logs.float()) # (k * d, d_state)
        Ds = self.Ds.float() # (k * d)
        dt_projs_bias = self.dt_projs_bias.float().view(-1) # (k * d)

        # assert len(xs.shape) == 3 and len(dts.shape) == 3 and len(Bs.shape) == 4 and len(Cs.shape) == 4
        # assert len(As.shape) == 2 and len(Ds.shape) == 1 and len(dt_projs_bias.shape) == 1

        out_y = selective_scan(
            xs, dts, 
            As, Bs, Cs, Ds, #z=None,
            delta_bias=dt_projs_bias,
            delta_softplus=True,
            # return_last_state=False,
        ).view(B, K, -1, L)
        # assert out_y.dtype == torch.float

        # inv_y = torch.flip(out_y[:, 2:4], dims=[-1]).view(B, 2, -1, L)
        # wh_y = torch.transpose(out_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)
        # invwh_y = torch.transpose(inv_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)
        # y = out_y[:, 0] + inv_y[:, 0] + wh_y + invwh_y
        # y = torch.transpose(y, dim0=1, dim1=2).contiguous().view(B, H, W, -1)
        y = torch.transpose(out_y[:, 0], dim0=1, dim1=2).contiguous().view(B, H, W, -1)
        y = self.out_norm(y)

        return y
    
    def forward_corev0_seq(self, x: torch.Tensor):
        selective_scan = selective_scan_fn

        B, C, H, W = x.shape
        L = H * W
        K = 4

        x_hwwh = torch.stack([x.view(B, -1, L), torch.transpose(x, dim0=2, dim1=3).contiguous().view(B, -1, L)], dim=1).view(B, 2, -1, L)
        xs = torch.cat([x_hwwh, torch.flip(x_hwwh, dims=[-1])], dim=1) # (b, k, d, l)

        x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs.view(B, K, -1, L), self.x_proj_weight)
        # x_dbl = x_dbl + self.x_proj_bias.view(1, K, -1, 1)
        dts, Bs, Cs = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=2)
        dts = torch.einsum("b k r l, k d r -> b k d l", dts.view(B, K, -1, L), self.dt_projs_weight)

        xs = xs.float() # (b, k, d, l)
        dts = dts.contiguous().float() # (b, k, d, l)
        Bs = Bs.float() # (b, k, d_state, l)
        Cs = Cs.float() # (b, k, d_state, l)
        
        As = -torch.exp(self.A_logs.float()).view(K, -1, self.d_state)  # (k, d, d_state)
        Ds = self.Ds.float().view(K, -1) # (k, d)
        dt_projs_bias = self.dt_projs_bias.float().view(K, -1) # (k, d)

        # assert len(xs.shape) == 4 and len(dts.shape) == 4 and len(Bs.shape) == 4 and len(Cs.shape) == 4
        # assert len(As.shape) == 3 and len(Ds.shape) == 2 and len(dt_projs_bias.shape) == 2

        out_y = []
        for i in range(4):
            yi = selective_scan(
                xs[:, i], dts[:, i], 
                As[i], Bs[:, i], Cs[:, i], Ds[i],
                delta_bias=dt_projs_bias[i],
                delta_softplus=True,
            ).view(B, -1, L)
            out_y.append(yi)
        out_y = torch.stack(out_y, dim=1)
        assert out_y.dtype == torch.float

        inv_y = torch.flip(out_y[:, 2:4], dims=[-1]).view(B, 2, -1, L)
        wh_y = torch.transpose(out_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)
        invwh_y = torch.transpose(inv_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)
        y = out_y[:, 0] + inv_y[:, 0] + wh_y + invwh_y
        y = torch.transpose(y, dim0=1, dim1=2).contiguous().view(B, H, W, -1)
        y = self.out_norm(y)

        return y

    def forward_corev1(self, x: torch.Tensor, float32=True):
        # float32 should be true in training!!!! otherwise, the output of selective_scan would be inf...
        selective_scan = selective_scan_fn_v1

        B, C, H, W = x.shape
        L = H * W

        xs = torch.stack([x.flatten(2, 3), x.transpose(dim0=2, dim1=3).contiguous().flatten(2, 3)], dim=1)
        xs = torch.cat([xs, torch.flip(xs, dims=[-1])], dim=1) # (b, k, d, l)

        x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs, self.x_proj_weight)
        # x_dbl = x_dbl + self.x_proj_bias.view(1, K, -1, 1)
        dts, Bs, Cs = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=2)
        dts = torch.einsum("b k r l, k d r -> b k d l", dts, self.dt_projs_weight)

        xs = xs.view(B, -1, L) # (b, k * d, l)
        dts = dts.contiguous().view(B, -1, L) # (b, k * d, l)
        As = -torch.exp(self.A_logs.to(torch.float))  # (k * d, d_state)
        Ds = self.Ds.to(torch.float) # (k * d)
        dt_projs_bias = self.dt_projs_bias.to(torch.float).view(-1) # (k * d)
        
        if float32:
            ys: torch.Tensor = selective_scan(
                xs.to(torch.float), 
                dts.to(torch.float), 
                As, 
                Bs.to(torch.float), 
                Cs.to(torch.float), 
                Ds,
                delta_bias=dt_projs_bias,
                delta_softplus=True,
            ).view(B, 4, -1, L)
            ys = ys[:, 0:2] + ys[:, 2:4].flip(dims=[-1]).view(B, 2, -1, L)
            y = ys[:, 0] + ys[:, 1].view(B, -1, W, H).transpose(dim0=2, dim1=3).contiguous().view(B, -1, L)
        else:
            out_y: torch.Tensor = selective_scan(
                xs, dts, 
                As, Bs, Cs, Ds,
                delta_bias=dt_projs_bias,
                delta_softplus=True,
            ).view(B, 4, -1, L)
            # assert out_y.dtype == torch.float16

            inv_y = torch.flip(out_y[:, 2:4], dims=[-1]).view(B, 2, -1, L)
            wh_y = torch.transpose(out_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)
            invwh_y = torch.transpose(inv_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)
            y = out_y[:, 0].float() + inv_y[:, 0].float() + wh_y.float() + invwh_y.float()
            
        if self.softmax_version:
            y = torch.softmax(y, dim=-1).to(x.dtype)
            y = torch.transpose(y, dim0=1, dim1=2).contiguous().view(B, H, W, -1)
        else:
            y = torch.transpose(y, dim0=1, dim1=2).contiguous().view(B, H, W, -1)
            y = self.out_norm(y).to(x.dtype)

            # if torch.isinf(y).any() or torch.isnan(y).any():
            #     for item in [y, xs, dts, As, Bs, Cs, Ds]:
            #         print(torch.isinf(item).any(), torch.isnan(item).any(), item.max(), item.min())
            #     import time; time.sleep(10000)
        
        return y

    def forward_corev1_share_ssm(self, x: torch.Tensor):
        selective_scan = selective_scan_fn_v1

        B, C, H, W = x.shape
        L = H * W

        def cross_scan_2d(x):
            # (B, C, H, W) => (B, K, C, H * W) with K = len([HW, WH, FHW, FWH])
            x_hwwh = torch.stack([x.flatten(2, 3), x.transpose(dim0=2, dim1=3).contiguous().flatten(2, 3)], dim=1)
            xs = torch.cat([x_hwwh, torch.flip(x_hwwh, dims=[-1])], dim=1) # (b, k, d, l)
            return xs
        
        x_dbl = torch.einsum("b d l, c d -> b c l", x.view(B, -1, L), self.x_proj_weight[0])
        # x_dbl = x_dbl + self.x_proj_bias.view(1, -1, 1)
        dt, BC = torch.split(x_dbl, [self.dt_rank, 2 * self.d_state], dim=1)
        dt = torch.einsum("b r l, d r -> b d l", dt, self.dt_projs_weight[0])
        x_dt_BC = torch.cat([x, dt.view(B, -1, H, W), BC.view(B, -1, H, W)], dim=1) # (b, -1, h, w)

        x_dt_BCs = cross_scan_2d(x_dt_BC) # (b, k, d, l)
        xs, dts, Bs, Cs = torch.split(x_dt_BCs, [self.d_inner, self.d_inner, self.d_state, self.d_state], dim=2)

        xs = xs.contiguous().view(B, -1, L) # (b, k * d, l)
        dts = dts.contiguous().view(B, -1, L) # (b, k * d, l)
        As = -torch.exp(self.A_logs.float()).repeat(4, 1) # (k * d, d_state)
        Ds = self.Ds.repeat(4) # (k * d)
        dt_projs_bias = self.dt_projs_bias.view(-1).repeat(4) # (k * d)

        # assert len(xs.shape) == 3 and len(dts.shape) == 3 and len(Bs.shape) == 4 and len(Cs.shape) == 4
        # assert len(As.shape) == 2 and len(Ds.shape) == 1 and len(dt_projs_bias.shape) == 1

        out_y = selective_scan(
            xs, dts, 
            As, Bs, Cs, Ds,
            delta_bias=dt_projs_bias,
            delta_softplus=True,
        ).view(B, 4, -1, L)
        # assert out_y.dtype == torch.float16

        inv_y = torch.flip(out_y[:, 2:4], dims=[-1]).view(B, 2, -1, L)
        wh_y = torch.transpose(out_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)
        invwh_y = torch.transpose(inv_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)
        y = out_y[:, 0].float() + inv_y[:, 0].float() + wh_y.float() + invwh_y.float()
        
        if self.softmax_version:
            y = torch.softmax(y, dim=-1).to(x.dtype)
            y = torch.transpose(y, dim0=1, dim1=2).contiguous().view(B, H, W, -1)
        else:
            y = torch.transpose(y, dim0=1, dim1=2).contiguous().view(B, H, W, -1)
            y = self.out_norm(y).to(x.dtype)
        
        return y

    def forward_corev1_share_a(self, x: torch.Tensor):
        selective_scan = selective_scan_fn_v1

        B, C, H, W = x.shape
        L = H * W

        def cross_scan_2d(x, dim=1):
            # (B, C, H, W) => (B, K, C, H * W) with K = len([HW, WH, FHW, FWH])
            x_hwwh = torch.stack([x.flatten(2, 3), x.transpose(dim0=2, dim1=3).contiguous().flatten(2, 3)], dim=dim)
            xs = torch.cat([x_hwwh, torch.flip(x_hwwh, dims=[-1])], dim=dim) # (b, k, d, l)
            return xs
        
        K = 4
        xs = cross_scan_2d(x, dim=1) # (b, d, k, l)

        x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs, self.x_proj_weight)
        # x_dbl = x_dbl + self.x_proj_bias.view(1, K, -1, 1)
        dts, Bs, Cs = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=2)
        dts = torch.einsum("b k r l, k d r -> b k d l", dts, self.dt_projs_weight)
        dts = dts + self.dt_projs_bias.to(xs.dtype).view(1, K, -1, 1)

        xs = xs.transpose(dim0=1, dim1=2).contiguous().view(B, -1, K * L)
        dts = dts.transpose(dim0=1, dim1=2).contiguous().view(B, -1, K * L)
        As = -torch.exp(self.A_logs.float()) # (D, N)
        Ds = self.Ds.view(-1) # (D)
        Bs = Bs.transpose(dim0=1, dim1=2).contiguous().view(B, 1, -1, K * L)
        Cs = Cs.transpose(dim0=1, dim1=2).contiguous().view(B, 1, -1, K * L)

        # assert len(xs.shape) == 3 and len(dts.shape) == 3 and len(Bs.shape) == 4 and len(Cs.shape) == 4
        # assert len(As.shape) == 2 and len(Ds.shape) == 1 and len(dt_projs_bias.shape) == 1
        # print(self.Ds.dtype, self.A_logs.dtype, self.dt_projs_bias.dtype, flush=True) # fp16, fp16, fp16

        out_y = selective_scan(
            xs, dts, 
            As, Bs, Cs, Ds,
            delta_bias=None,
            delta_softplus=True,
        ).view(B, -1, 4, L)
        # assert out_y.dtype == torch.float16

        inv_y = torch.flip(out_y[:, :, 2:4], dims=[-1]).view(B, -1, 2, L)
        wh_y = torch.transpose(out_y[:, :, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)
        invwh_y = torch.transpose(inv_y[:, :, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)
        y = out_y[:, :, 0].float() + inv_y[:, :, 0].float() + wh_y.float() + invwh_y.float()
        
        if self.softmax_version:
            y = torch.softmax(y, dim=-1).to(x.dtype)
            y = torch.transpose(y, dim0=1, dim1=2).contiguous().view(B, H, W, -1)
        else:
            y = torch.transpose(y, dim0=1, dim1=2).contiguous().view(B, H, W, -1)
            y = self.out_norm(y).to(x.dtype)
        
        return y

    def forward_corev2(self, x: torch.Tensor, nrows=-1):
        return cross_selective_scan(
            x, self.x_proj_weight, None, self.dt_projs_weight, self.dt_projs_bias,
            self.A_logs, self.Ds, getattr(self, "out_norm", None), self.softmax_version, 
            nrows=nrows,
        )
        
    def forward_core_1d(self, x: torch.Tensor):
        return selective_scan_1d(
            x, self.x_proj_weight, None, self.dt_projs_weight, self.dt_projs_bias,
            self.A_logs, self.Ds, getattr(self, "out_norm", None), self.softmax_version,
        )
    
    # forward_core = forward_core_share_ssm
    # forward_core = forward_core_share_a
    # forward_core = forward_corev1
    forward_core = forward_corev2 # vmamba
    # forward_core = forward_corev0 # ori mamba
    # forward_core = forward_core_1d # ori mamba

    def forward(self, x: torch.Tensor, **kwargs):
        xz = self.in_proj(x)
        if self.d_conv > 1:
            x, z = xz.chunk(2, dim=-1) # (b, h, w, d)
            x = x.permute(0, 3, 1, 2).contiguous()
            x = self.act(self.conv2d(x)) # (b, d, h, w)
            y = self.forward_core(x)
            if self.softmax_version:
                y = y * z
            else:
                y = y * F.silu(z)
        else:
            if self.softmax_version:
                x, z = xz.chunk(2, dim=-1) # (b, h, w, d)
                x = F.silu(x)
            else:
                xz = F.silu(xz)
                x, z = xz.chunk(2, dim=-1) # (b, h, w, d)
            x = x.permute(0, 3, 1, 2).contiguous()
            y = self.forward_core(x)
            y = y * z
        out = self.dropout(self.out_proj(y))
        return out


class MM_SS2D(nn.Module):
    '''
    Multimodal Mamba Selective Scan 2D
    '''
    def __init__(
        self,
        # basic dims ===========
        d_model=96,
        d_state=4,
        ssm_ratio=2,
        dt_rank="auto",
        # dwconv ===============
        # d_conv=-1, # < 2 means no conv 
        d_conv=3, # < 2 means no conv 
        conv_bias=True,
        # ======================
        dropout=0.,
        bias=False,
        # dt init ==============
        dt_min=0.001,
        dt_max=0.1,
        dt_init="random",
        dt_scale=1.0,
        dt_init_floor=1e-4,
        # ======================
        softmax_version=False,
        # ======================
        **kwargs,
    ):
        if DEV:
            d_conv = -1
            
        factory_kwargs = {"device": None, "dtype": None}
        super().__init__()
        self.softmax_version = softmax_version
        self.d_model = d_model
        self.d_state = math.ceil(self.d_model / 6) if d_state == "auto" else d_state # 20240109
        self.d_conv = d_conv
        self.expand = ssm_ratio
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank

        self.in_proj = nn.Linear(self.d_model, self.d_inner, bias=bias, **factory_kwargs)
        self.in_proj_modalx = nn.Linear(self.d_model, self.d_inner, bias=bias, **factory_kwargs)
        
        # conv =======================================
        if self.d_conv > 1:
            self.conv2d = nn.Conv2d(
                in_channels=self.d_inner,
                out_channels=self.d_inner,
                groups=self.d_inner,
                bias=conv_bias,
                kernel_size=d_conv,
                padding=(d_conv - 1) // 2,
                **factory_kwargs,
            )
            self.conv2d_modalx = nn.Conv2d(
                in_channels=self.d_inner,
                out_channels=self.d_inner,
                groups=self.d_inner,
                bias=conv_bias,
                kernel_size=d_conv,
                padding=(d_conv - 1) // 2,
                **factory_kwargs,
            )
            self.act = nn.SiLU()

        # x proj; dt proj ============================
        self.K = 2
        self.x_proj = [
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs)
            for _ in range(self.K)
        ]
        self.x_proj_weight = nn.Parameter(torch.stack([t.weight for t in self.x_proj], dim=0)) # (K, N, inner)
        del self.x_proj

        self.dt_projs = [
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs)
            for _ in range(self.K)
        ]
        self.dt_projs_weight = nn.Parameter(torch.stack([t.weight for t in self.dt_projs], dim=0)) # (K, inner, rank)
        self.dt_projs_bias = nn.Parameter(torch.stack([t.bias for t in self.dt_projs], dim=0)) # (K, inner)
        del self.dt_projs
        
        # A, D =======================================
        self.K2 = self.K
        self.A_logs = self.A_log_init(self.d_state, self.d_inner, copies=self.K2, merge=True) # (K * D, N)
        self.Ds = self.D_init(self.d_inner, copies=self.K2, merge=True) # (K * D)

        # out proj =======================================
        if not self.softmax_version:
            self.out_norm1 = nn.LayerNorm(self.d_inner)
            self.out_norm2 = nn.LayerNorm(self.d_inner)
        self.out_proj = nn.Linear(self.d_inner*2, self.d_model, bias=bias, **factory_kwargs)
        self.dropout = nn.Dropout(dropout) if dropout > 0. else nn.Identity()
        
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Sequential(
            nn.Linear(self.d_inner, self.d_inner // 16, bias=False),
            nn.SiLU(inplace=True),
            nn.Linear(self.d_inner // 16, self.d_inner, bias=False),
            nn.Sigmoid(),
        )
        self.fc2 = nn.Sequential(
            nn.Linear(self.d_inner, self.d_inner // 16, bias=False),
            nn.SiLU(inplace=True),
            nn.Linear(self.d_inner // 16, self.d_inner, bias=False),
            nn.Sigmoid(),
        )

    @staticmethod
    def dt_init(dt_rank, d_inner, dt_scale=1.0, dt_init="random", dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4, **factory_kwargs):
        dt_proj = nn.Linear(dt_rank, d_inner, bias=True, **factory_kwargs)

        # Initialize special dt projection to preserve variance at initialization
        dt_init_std = dt_rank**-0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        # Initialize dt bias so that F.softplus(dt_bias) is between dt_min and dt_max
        dt = torch.exp(
            torch.rand(d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            dt_proj.bias.copy_(inv_dt)
        # Our initialization would set all Linear.bias to zero, need to mark this one as _no_reinit
        # dt_proj.bias._no_reinit = True
        
        return dt_proj

    @staticmethod
    def A_log_init(d_state, d_inner, copies=-1, device=None, merge=True):
        # S4D real initialization
        A = repeat(
            torch.arange(1, d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=d_inner,
        ).contiguous()
        A_log = torch.log(A)  # Keep A_log in fp32
        if copies > 0:
            A_log = repeat(A_log, "d n -> r d n", r=copies)
            if merge:
                A_log = A_log.flatten(0, 1)
        A_log = nn.Parameter(A_log)
        A_log._no_weight_decay = True
        return A_log

    @staticmethod
    def D_init(d_inner, copies=-1, device=None, merge=True):
        # D "skip" parameter
        D = torch.ones(d_inner, device=device)
        if copies > 0:
            D = repeat(D, "n1 -> r n1", r=copies)
            if merge:
                D = D.flatten(0, 1)
        D = nn.Parameter(D)  # Keep in fp32
        D._no_weight_decay = True
        return D
     
    
    # def forward_core_orimamba(self, x_rgb: torch.Tensor, x_e: torch.Tensor):
        selective_scan = selective_scan_fn

        B, D, H, W = x_rgb.shape
        x_dbl = self.x_proj(rearrange(x, "b d l -> (b l) d"))  # (bl d)
        dt, B, C = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1)
        dt = self.dt_proj.weight @ dt.t()
        dt = rearrange(dt, "d (b l) -> b d l", l=L)
        A = -torch.exp(self.A_log.float())  # (k * d, d_state)
        B = rearrange(B, "(b l) dstate -> b dstate l", l=L).contiguous()
        C = rearrange(C, "(b l) dstate -> b dstate l", l=L).contiguous()

        y = selective_scan(
            x, dt,
            A, B, C, self.D.float(),
            delta_bias=self.dt_proj.bias.float(),
            delta_softplus=True,
        )
        # assert out_y.dtype == torch.float
        y = rearrange(y, "b d l -> b l d")
        return y
    
    def forward_corev2_multimodal(self, x_rgb: torch.Tensor, x_e: torch.Tensor, nrows=-1):
        return cross_selective_scan_multimodal_k2(
            x_rgb, x_e, self.x_proj_weight, None, self.dt_projs_weight, self.dt_projs_bias,
            self.A_logs, self.Ds, getattr(self, "out_norm1", None), getattr(self, "out_norm2", None), self.softmax_version, 
            nrows=nrows,
        )

    def forward(self, x_rgb: torch.Tensor, x_e: torch.Tensor):
        x_rgb = self.in_proj(x_rgb)
        x_e = self.in_proj_modalx(x_e)
        if self.d_conv > 1:
            x_rgb_trans = x_rgb.permute(0, 3, 1, 2).contiguous()
            x_e_trans = x_e.permute(0, 3, 1, 2).contiguous()
            x_rgb_conv = self.act(self.conv2d(x_rgb_trans)) # (b, d, h, w)
            x_e_conv = self.act(self.conv2d_modalx(x_e_trans)) # (b, d, h, w)
            y_rgb, y_e = self.forward_corev2_multimodal(x_rgb_conv, x_e_conv) # b, d, h, w -> b, h, w, d
            # SE to get attention, scale
            b, d, h, w = x_rgb_trans.shape
            x_rgb_squeeze = self.avg_pool(x_rgb_trans).view(b, d)
            x_e_squeeze = self.avg_pool(x_e_trans).view(b, d)
            x_rgb_exitation = self.fc1(x_rgb_squeeze).view(b, d, 1, 1).permute(0, 2, 3, 1).contiguous() # b, 1, 1, d
            x_e_exitation = self.fc2(x_e_squeeze).view(b, d, 1, 1).permute(0, 2, 3, 1).contiguous() 
            y_rgb = y_rgb * x_e_exitation
            y_e = y_e * x_rgb_exitation
            y = torch.concat([y_rgb, y_e], dim=-1)
        out = self.dropout(self.out_proj(y))
        return out

# =====================================================
class SSM(nn.Module):
    def __init__(
            self,
            # basic dims ===========
            d_model=96,
            d_state=4,
            ssm_ratio=2,
            dt_rank="auto",
            # dt init ==============
            dt_min=0.001,
            dt_max=0.1,
            dt_init="random",
            dt_scale=1.0,
            dt_init_floor=1e-4,
            # ======================
            **kwargs,
    ):
        factory_kwargs = {"device": None, "dtype": None}
        super().__init__()
        self.d_model = d_model
        self.d_state = math.ceil(self.d_model / 6) if d_state == "auto" else d_state  # 20240109
        self.expand = ssm_ratio
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank

        # x proj; dt proj ============================
        self.x_proj = nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs)

        self.dt_proj = self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                                    **factory_kwargs)

        # A, D =======================================
        self.A_log = self.A_log_init(self.d_state, self.d_inner)  # (D, N)
        self.D = self.D_init(self.d_inner)  # (D)

        # out norm ===================================
        self.out_norm = nn.LayerNorm(self.d_inner)

    @staticmethod
    def dt_init(dt_rank, d_inner, dt_scale=1.0, dt_init="random", dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4,
                **factory_kwargs):
        dt_proj = nn.Linear(dt_rank, d_inner, bias=True, **factory_kwargs)

        # Initialize special dt projection to preserve variance at initialization
        dt_init_std = dt_rank ** -0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        # Initialize dt bias so that F.softplus(dt_bias) is between dt_min and dt_max
        dt = torch.exp(
            torch.rand(d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            dt_proj.bias.copy_(inv_dt)
        # Our initialization would set all Linear.bias to zero, need to mark this one as _no_reinit
        # dt_proj.bias._no_reinit = True

        return dt_proj

    @staticmethod
    def A_log_init(d_state, d_inner, copies=-1, device=None, merge=True):
        # S4D real initialization
        A = repeat(
            torch.arange(1, d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=d_inner,
        ).contiguous()
        A_log = torch.log(A)  # Keep A_log in fp32
        if copies > 0:
            A_log = repeat(A_log, "d n -> r d n", r=copies)
            if merge:
                A_log = A_log.flatten(0, 1)
        A_log = nn.Parameter(A_log)
        A_log._no_weight_decay = True
        return A_log

    @staticmethod
    def D_init(d_inner, copies=-1, device=None, merge=True):
        # D "skip" parameter
        D = torch.ones(d_inner, device=device)
        if copies > 0:
            D = repeat(D, "n1 -> r n1", r=copies)
            if merge:
                D = D.flatten(0, 1)
        D = nn.Parameter(D)  # Keep in fp32
        D._no_weight_decay = True
        return D

    def forward(self, x: torch.Tensor):
        selective_scan = selective_scan_fn_v1
        B, L, d = x.shape
        x = x.permute(0, 2, 1)
        x_dbl = self.x_proj(rearrange(x, "b d l -> (b l) d"))  # (bl d)
        dt, B, C = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1)
        dt = self.dt_proj.weight @ dt.t()
        dt = rearrange(dt, "d (b l) -> b d l", l=L)
        A = -torch.exp(self.A_log.float())  # (k * d, d_state)
        B = rearrange(B, "(b l) dstate -> b dstate l", l=L).contiguous()
        C = rearrange(C, "(b l) dstate -> b dstate l", l=L).contiguous()

        y = selective_scan(
            x, dt,
            A, B, C, self.D.float(),
            delta_bias=self.dt_proj.bias.float(),
            delta_softplus=True,
        )
        # assert out_y.dtype == torch.float
        y = rearrange(y, "b d l -> b l d")
        y = self.out_norm(y)
        return y
    
    
# =====================================================
class Cross_Mamba_Attention_SSM(nn.Module):
    def __init__(
            self,
            # basic dims ===========
            d_model=96,
            d_state=4,
            ssm_ratio=2,
            dt_rank="auto",
            # dt init ==============
            dt_min=0.001,
            dt_max=0.1,
            dt_init="random",
            dt_scale=1.0,
            dt_init_floor=1e-4,
            # ======================
            **kwargs,
    ):
        factory_kwargs = {"device": None, "dtype": None}
        super().__init__()
        self.d_model = d_model
        self.d_state = math.ceil(self.d_model / 6) if d_state == "auto" else d_state  # 20240109
        self.expand = ssm_ratio
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank

        # x proj; dt proj ============================
        self.x_proj_1 = nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs)
        self.x_proj_2 = nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs)

        self.dt_proj_1 = self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                                    **factory_kwargs)
        self.dt_proj_2 = self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                                    **factory_kwargs)

        # A, D =======================================
        self.A_log_1 = self.A_log_init(self.d_state, self.d_inner)  # (D, N)
        self.A_log_2 = self.A_log_init(self.d_state, self.d_inner)  # (D)
        self.D_1 = self.D_init(self.d_inner)  # (D)
        self.D_2 = self.D_init(self.d_inner)  # (D)

        # out norm ===================================
        self.out_norm_1 = nn.LayerNorm(self.d_inner)
        self.out_norm_2 = nn.LayerNorm(self.d_inner)

    @staticmethod
    def dt_init(dt_rank, d_inner, dt_scale=1.0, dt_init="random", dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4,
                **factory_kwargs):
        dt_proj = nn.Linear(dt_rank, d_inner, bias=True, **factory_kwargs)

        # Initialize special dt projection to preserve variance at initialization
        dt_init_std = dt_rank ** -0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        # Initialize dt bias so that F.softplus(dt_bias) is between dt_min and dt_max
        dt = torch.exp(
            torch.rand(d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            dt_proj.bias.copy_(inv_dt)
        # Our initialization would set all Linear.bias to zero, need to mark this one as _no_reinit
        # dt_proj.bias._no_reinit = True

        return dt_proj

    @staticmethod
    def A_log_init(d_state, d_inner, copies=-1, device=None, merge=True):
        # S4D real initialization
        A = repeat(
            torch.arange(1, d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=d_inner,
        ).contiguous()
        A_log = torch.log(A)  # Keep A_log in fp32
        if copies > 0:
            A_log = repeat(A_log, "d n -> r d n", r=copies)
            if merge:
                A_log = A_log.flatten(0, 1)
        A_log = nn.Parameter(A_log)
        A_log._no_weight_decay = True
        return A_log

    @staticmethod
    def D_init(d_inner, copies=-1, device=None, merge=True):
        # D "skip" parameter
        D = torch.ones(d_inner, device=device)
        if copies > 0:
            D = repeat(D, "n1 -> r n1", r=copies)
            if merge:
                D = D.flatten(0, 1)
        D = nn.Parameter(D)  # Keep in fp32
        D._no_weight_decay = True
        return D

    def forward(self, x_rgb: torch.Tensor, x_e: torch.Tensor):
        selective_scan = selective_scan_fn_v1
        B, L, d = x_rgb.shape
        x_rgb = x_rgb.permute(0, 2, 1)
        x_e = x_e.permute(0, 2, 1)
        x_dbl_rgb = self.x_proj_1(rearrange(x_rgb, "b d l -> (b l) d"))  # (bl d)
        x_dbl_e = self.x_proj_2(rearrange(x_e, "b d l -> (b l) d"))  # (bl d)
        dt_rgb, B_rgb, C_rgb = torch.split(x_dbl_rgb, [self.dt_rank, self.d_state, self.d_state], dim=-1)
        dt_e, B_e, C_e = torch.split(x_dbl_e, [self.dt_rank, self.d_state, self.d_state], dim=-1)
        dt_rgb = self.dt_proj_1.weight @ dt_rgb.t()
        dt_e = self.dt_proj_2.weight @ dt_e.t()
        dt_rgb = rearrange(dt_rgb, "d (b l) -> b d l", l=L)
        dt_e = rearrange(dt_e, "d (b l) -> b d l", l=L)
        A_rgb = -torch.exp(self.A_log_1.float())  # (k * d, d_state)
        A_e = -torch.exp(self.A_log_2.float())  # (k * d, d_state)
        B_rgb = rearrange(B_rgb, "(b l) dstate -> b dstate l", l=L).contiguous()
        B_e = rearrange(B_e, "(b l) dstate -> b dstate l", l=L).contiguous()
        C_rgb = rearrange(C_rgb, "(b l) dstate -> b dstate l", l=L).contiguous()
        C_e = rearrange(C_e, "(b l) dstate -> b dstate l", l=L).contiguous()

        y_rgb = selective_scan(
            x_rgb, dt_rgb,
            A_rgb, B_rgb, C_e, self.D_1.float(),
            delta_bias=self.dt_proj_1.bias.float(),
            delta_softplus=True,
        )
        y_e = selective_scan(
            x_e, dt_e,
            A_e, B_e, C_rgb, self.D_2.float(),
            delta_bias=self.dt_proj_2.bias.float(),
            delta_softplus=True,
        )
        # assert out_y.dtype == torch.float
        y_rgb = rearrange(y_rgb, "b d l -> b l d")
        y_rgb = self.out_norm_1(y_rgb)
        y_e = rearrange(y_e, "b d l -> b l d")
        y_e = self.out_norm_2(y_e)
        return y_rgb, y_e


class CAMF_SS2D_SSM(nn.Module):
    '''
    Channel Attention Mamba Fusion Selective Scan 2D Module with SSM
    '''
    def __init__(
        self,
        # basic dims ===========
        d_model=96,
        d_state=16,
        ssm_ratio=2,
        dt_rank="auto",
        # dwconv ===============
        # d_conv=-1, # < 2 means no conv 
        d_conv=3, # < 2 means no conv 
        conv_bias=True,
        # ======================
        dropout=0.,
        bias=False,
        # dt init ==============
        dt_min=0.001,
        dt_max=0.1,
        dt_init="random",
        dt_scale=1.0,
        dt_init_floor=1e-4,
        # ======================
        softmax_version=False,
        # ======================
        **kwargs,
    ):            
        factory_kwargs = {"device": None, "dtype": None}
        super().__init__()
        self.softmax_version = softmax_version
        self.d_model = d_model
        self.d_state = math.ceil(self.d_model / 6) if d_state == "auto" else d_state # 20240109
        self.d_conv = d_conv
        self.expand = ssm_ratio
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank

        self.in_proj = nn.Linear(self.d_model, self.d_inner, bias=bias, **factory_kwargs)
        self.in_proj_modalx = nn.Linear(self.d_model, self.d_inner, bias=bias, **factory_kwargs)
        
        # conv =======================================
        if self.d_conv > 1:
            self.conv2d = nn.Conv2d(
                in_channels=self.d_inner,
                out_channels=self.d_inner,
                groups=self.d_inner,
                bias=conv_bias,
                kernel_size=d_conv,
                padding=(d_conv - 1) // 2,
                **factory_kwargs,
            )
            self.act = nn.SiLU()

        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)
        self.dropout = nn.Dropout(dropout) if dropout > 0. else nn.Identity()
        
        # channel attention
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(self.d_inner, self.d_inner // 16, bias=False),
            nn.SiLU(inplace=True),
            nn.Linear(self.d_inner // 16, self.d_inner, bias=False),
            nn.Sigmoid(),
        )
        self.ssm = SSM(
            d_model=self.d_model,
            d_state=self.d_state,
            ssm_ratio=ssm_ratio,
            dt_rank=dt_rank,
            dt_min=dt_min,
            dt_max=dt_max,
            dt_init=dt_init,
            dt_scale=dt_scale,
            dt_init_floor=dt_init_floor,
            **kwargs,
        )

    def forward(self, x_rgb: torch.Tensor, x_e: torch.Tensor):
        x_rgb = self.in_proj(x_rgb)
        x_e = self.in_proj_modalx(x_e)
        B, H, W, D = x_rgb.shape
        if self.d_conv > 1:
            x_rgb_trans = x_rgb.permute(0, 3, 1, 2).contiguous()
            x_e_trans = x_e.permute(0, 3, 1, 2).contiguous()
            x_rgb_conv = self.act(self.conv2d(x_rgb_trans)) # (b, d, h, w)
            x_e_conv = self.act(self.conv2d(x_e_trans)) # (b, d, h, w)
            x_rgb_conv = rearrange(x_rgb_conv, "b d h w -> b (h w) d")
            x_e_conv = rearrange(x_e_conv, "b d h w -> b (h w) d")
            x_concat_L = torch.cat([x_rgb_conv, x_e_conv], dim=1) # b, 2L, d
            y = self.ssm(x_concat_L) # b, 2L, d -> b, 2L, d
            B, L, d = y.shape
            y_rgb = y[:, :L//2] # b, L, d
            y_e = y[:, L//2:]
            # SE to get attention
            b, d, h, w = x_rgb_trans.shape
            x_rgb_squeeze = self.avg_pool(x_rgb_trans).view(b, d)
            x_e_squeeze = self.avg_pool(x_e_trans).view(b, d)
            x_rgb_exitation = self.fc(x_rgb_squeeze).view(b, d, 1).permute(0, 2, 1).contiguous() # b, 1, d
            x_e_exitation = self.fc(x_e_squeeze).view(b, d, 1).permute(0, 2, 1).contiguous() 
            y_rgb = y_rgb * x_rgb_exitation.expand_as(y_rgb)
            y_e = y_e * x_e_exitation.expand_as(y_e)
            y = y_rgb + y_e # b, L, d
            # to b, d, h, w
            y = y.view(B, H, W, -1)
            # if self.softmax_version:
            #     y = y * x_rgb * x_e
            # else:
            #     y = y * F.silu(x_rgb) * F.silu(x_e)
        # else:
        #     x_rgb = F.silu(x_rgb)
        #     x_e = F.silu(x_e)
        #     x_rgb_trans = x_rgb.permute(0, 3, 1, 2).contiguous()
        #     x_e_trans = x_e.permute(0, 3, 1, 2).contiguous()
        #     y = self.forward_corev2_multimodal(x_rgb_trans, x_e_trans)
        #     y = y * x_rgb * x_e
        out = self.dropout(self.out_proj(y))
        return out


class CrossMambaFusion_SS2D_SSM(nn.Module):
    '''
    Cross Mamba Attention Fusion Selective Scan 2D Module with SSM
    '''
    def __init__(
        self,
        # basic dims ===========
        d_model=96,
        d_state=16,
        ssm_ratio=2,
        dt_rank="auto",
        # dwconv ===============
        # d_conv=-1, # < 2 means no conv 
        d_conv=3, # < 2 means no conv 
        conv_bias=True,
        # ======================
        dropout=0.,
        bias=False,
        # dt init ==============
        dt_min=0.001,
        dt_max=0.1,
        dt_init="random",
        dt_scale=1.0,
        dt_init_floor=1e-4,
        # ======================
        softmax_version=False,
        # ======================
        **kwargs,
    ):            
        factory_kwargs = {"device": None, "dtype": None}
        super().__init__()
        self.softmax_version = softmax_version
        self.d_model = d_model
        self.d_state = math.ceil(self.d_model / 6) if d_state == "auto" else d_state # 20240109
        self.d_conv = d_conv
        self.expand = ssm_ratio
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank

        self.in_proj = nn.Linear(self.d_model, self.d_inner, bias=bias, **factory_kwargs)
        self.in_proj_modalx = nn.Linear(self.d_model, self.d_inner, bias=bias, **factory_kwargs)
        
        # conv =======================================
        if self.d_conv > 1:
            self.conv2d = nn.Conv2d(
                in_channels=self.d_inner,
                out_channels=self.d_inner,
                groups=self.d_inner,
                bias=conv_bias,
                kernel_size=d_conv,
                padding=(d_conv - 1) // 2,
                **factory_kwargs,
            )
            self.act = nn.SiLU()

        self.out_proj_rgb = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)
        self.out_proj_e = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)
        self.dropout_rgb = nn.Dropout(dropout) if dropout > 0. else nn.Identity()
        self.dropout_e = nn.Dropout(dropout) if dropout > 0. else nn.Identity()
        
        self.CMA_ssm = Cross_Mamba_Attention_SSM(
            d_model=self.d_model,
            d_state=self.d_state,
            ssm_ratio=ssm_ratio,
            dt_rank=dt_rank,
            dt_min=dt_min,
            dt_max=dt_max,
            dt_init=dt_init,
            dt_scale=dt_scale,
            dt_init_floor=dt_init_floor,
            **kwargs,
        )

    def forward(self, x_rgb: torch.Tensor, x_e: torch.Tensor):
        x_rgb = self.in_proj(x_rgb)
        x_e = self.in_proj_modalx(x_e)
        B, H, W, D = x_rgb.shape
        if self.d_conv > 1:
            x_rgb_trans = x_rgb.permute(0, 3, 1, 2).contiguous()
            x_e_trans = x_e.permute(0, 3, 1, 2).contiguous()
            x_rgb_conv = self.act(self.conv2d(x_rgb_trans)) # (b, d, h, w)
            x_e_conv = self.act(self.conv2d(x_e_trans)) # (b, d, h, w)
            x_rgb_conv = rearrange(x_rgb_conv, "b d h w -> b (h w) d")
            x_e_conv = rearrange(x_e_conv, "b d h w -> b (h w) d")
            y_rgb, y_e = self.CMA_ssm(x_rgb_conv, x_e_conv) 
            # to b, d, h, w
            y_rgb = y_rgb.view(B, H, W, -1)
            y_e = y_e.view(B, H, W, -1)
            
            # y_rgb = y_rgb * F.silu(x_rgb)
            # y_e = y_e * F.silu(x_e)
            
            # if self.softmax_version:
            #     y = y * x_rgb * x_e
            # else:
            #     y = y * F.silu(x_rgb) * F.silu(x_e)
        # else:
        #     x_rgb = F.silu(x_rgb)
        #     x_e = F.silu(x_e)
        #     x_rgb_trans = x_rgb.permute(0, 3, 1, 2).contiguous()
        #     x_e_trans = x_e.permute(0, 3, 1, 2).contiguous()
        #     y = self.forward_corev2_multimodal(x_rgb_trans, x_e_trans)
        #     y = y * x_rgb * x_e
        out_rgb = self.dropout_rgb(self.out_proj_rgb(y_rgb))
        out_e = self.dropout_e(self.out_proj_e(y_e))
        return out_rgb, out_e

class Permute(nn.Module):
    def __init__(self, *args):
        super().__init__()
        self.args = args

    def forward(self, x: torch.Tensor):
        return x.permute(*self.args)


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.,channels_first=False):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        Linear = partial(nn.Conv2d, kernel_size=1, padding=0) if channels_first else nn.Linear
        self.fc1 = Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class VSSBlock(nn.Module):
    def __init__(
        self,
        hidden_dim: int = 0,
        drop_path: float = 0,
        norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
        attn_drop_rate: float = 0,
        d_state: int = 16,
        dt_rank: Any = "auto",
        ssm_ratio=2.0,
        shared_ssm=False,
        softmax_version=False,
        use_checkpoint: bool = False,
        mlp_ratio=4.0,
        act_layer=nn.GELU,
        drop: float = 0.0,
        **kwargs,
    ):
        super().__init__()
        self.use_checkpoint = use_checkpoint
        self.norm = norm_layer(hidden_dim)
        self.op = SS2D(
            d_model=hidden_dim, 
            dropout=attn_drop_rate, 
            d_state=d_state, 
            ssm_ratio=ssm_ratio, 
            dt_rank=dt_rank,
            shared_ssm=shared_ssm,
            softmax_version=softmax_version,
            **kwargs
        )
        self.drop_path = DropPath(drop_path)
        
        self.mlp_branch = mlp_ratio > 0
        if self.mlp_branch:
            self.norm2 = norm_layer(hidden_dim)
            mlp_hidden_dim = int(hidden_dim * mlp_ratio)
            self.mlp = Mlp(in_features=hidden_dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop, channels_first=False)

    def _forward(self, input: torch.Tensor):
        x = input + self.drop_path(self.op(self.norm(input)))
        if self.mlp_branch:
            x = x + self.drop_path(self.mlp(self.norm2(x))) # FFN
        return x

    def forward(self, input: torch.Tensor):
        if self.use_checkpoint:
            return checkpoint.checkpoint(self._forward, input)
        else:
            return self._forward(input)

# class ASPPConv(nn.Sequential):
#     def __init__(self, in_channels, out_channels, dilation):
#         modules = [
#             nn.Conv2d(in_channels, out_channels, 3, padding=dilation, dilation=dilation, bias=False),
#             nn.BatchNorm2d(out_channels),
#             nn.ReLU()
#         ]
#         super(ASPPConv, self).__init__(*modules)
        
class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding, dilation):
        super(DepthwiseSeparableConv, self).__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, padding=padding, dilation=dilation, groups=in_channels, bias=False)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x

class ASPPConv(nn.Sequential):
    def __init__(self, in_channels, out_channels, dilation):
        modules = [
            DepthwiseSeparableConv(in_channels, out_channels, 3, padding=dilation, dilation=dilation),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        ]
        super(ASPPConv, self).__init__(*modules)

class ASPPPooling(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super(ASPPPooling, self).__init__(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU())

    def forward(self, x):
        size = x.shape[-2:]
        for mod in self:
            x = mod(x)
        return F.interpolate(x, size=size, mode='bilinear', align_corners=False)

class ASPP(nn.Module):
    def __init__(self, in_channels, atrous_rates=[6], out_channels=256):
        super(ASPP, self).__init__()
        modules = []
        modules.append(nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()))

        rates = tuple(atrous_rates)
        for rate in rates:
            modules.append(ASPPConv(in_channels, out_channels, rate))

        modules.append(ASPPPooling(in_channels, out_channels))

        self.convs = nn.ModuleList(modules)

        self.project = nn.Sequential(
            nn.Conv2d(len(self.convs) * out_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Dropout(0.5))

    def forward(self, x):
        res = []
        for conv in self.convs:
            res.append(conv(x))
        res = torch.cat(res, dim=1)
        return self.project(res)


class ChannelAttention(nn.Module):
    def __init__(self, num_feat, squeeze_factor=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1) 
        self.fc = nn.Sequential(
            nn.Conv2d(num_feat, num_feat // squeeze_factor, 1, bias=False),
            nn.SiLU(inplace=True),
            nn.Conv2d(num_feat // squeeze_factor, num_feat, 1, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        attn = avg_out + max_out
        return x * self.sigmoid(attn)


class ChannelAttentionBlock(nn.Module):

    def __init__(self, num_feat, compress_ratio=3, squeeze_factor=30):
        super(ChannelAttentionBlock, self).__init__()

        self.cab = nn.Sequential(
            nn.Conv2d(num_feat, num_feat // compress_ratio, 3, 1, 1),
            nn.GELU(),
            nn.Conv2d(num_feat // compress_ratio, num_feat, 3, 1, 1),
            ChannelAttention(num_feat, squeeze_factor)
            )

    def forward(self, x):
        return self.cab(x)


class VSSDecoderBlock(nn.Module):
    def __init__(
        self,
        hidden_dim: int = 0,
        drop_path: float = 0,
        norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
        attn_drop_rate: float = 0,
        d_state: int = 16,
        dt_rank: Any = "auto",
        ssm_ratio=2.0,
        shared_ssm=False,
        softmax_version=False,
        use_checkpoint: bool = False,
        mlp_ratio=4.0,
        act_layer=nn.GELU,
        drop: float = 0.0,
        **kwargs,
    ):
        super().__init__()
        self.use_checkpoint = use_checkpoint
        self.norm1 = norm_layer(hidden_dim)
        self.scale1 = nn.Parameter(torch.ones(hidden_dim))
        self.op = SS2D(
            d_model=hidden_dim, 
            dropout=attn_drop_rate, 
            d_state=d_state, 
            ssm_ratio=ssm_ratio, 
            dt_rank=dt_rank,
            shared_ssm=shared_ssm,
            softmax_version=softmax_version,
            **kwargs
        )
        self.drop_path = DropPath(drop_path)
        self.conv_blk = ChannelAttentionBlock(hidden_dim)
        self.norm2 = norm_layer(hidden_dim)
        self.scale2 = nn.Parameter(torch.ones(hidden_dim))
        
    def _forward(self, input: torch.Tensor):
        x = input*self.scale1 + self.drop_path(self.op(self.norm1(input)))

        y = self.conv_blk(self.norm2(x).permute(0, 3, 1, 2).contiguous()) + (x*self.scale2).permute(0, 3, 1, 2).contiguous()
        y = y.permute(0, 2, 3, 1).contiguous()
        return y

    def forward(self, input: torch.Tensor):
        if self.use_checkpoint:
            return checkpoint.checkpoint(self._forward, input)
        else:
            return self._forward(input)


class CAMFBlock(nn.Module):
    '''
    channel attention fusion block, with 1D SSM
    '''
    def __init__(
        self,
        hidden_dim: int = 0,
        drop_path: float = 0,
        norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
        attn_drop_rate: float = 0,
        d_state: int = 16,
        dt_rank: Any = "auto",
        ssm_ratio=2.0,
        shared_ssm=False,
        softmax_version=False,
        use_checkpoint: bool = False,
        mlp_ratio=0.0,
        act_layer=nn.GELU,
        drop: float = 0.0,
        **kwargs,
    ):
        super().__init__()
        self.use_checkpoint = use_checkpoint
        # self.norm = norm_layer(hidden_dim)
        self.op = CAMF_SS2D_SSM(
            d_model=hidden_dim, 
            dropout=attn_drop_rate, 
            d_state=d_state, 
            ssm_ratio=ssm_ratio, 
            dt_rank=dt_rank,
            shared_ssm=shared_ssm,
            softmax_version=softmax_version,
            **kwargs
        )
        self.drop_path = DropPath(drop_path)
        
        self.mlp_branch = mlp_ratio > 0
        if self.mlp_branch:
            self.norm2 = norm_layer(hidden_dim)
            mlp_hidden_dim = int(hidden_dim * mlp_ratio)
            self.mlp = Mlp(in_features=hidden_dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop, channels_first=False)

    def _forward(self, x_rgb: torch.Tensor, x_e: torch.Tensor):
        x = x_rgb + x_e + self.drop_path(self.op(x_rgb, x_e))
        if self.mlp_branch:
            x = x + self.drop_path(self.mlp(self.norm2(x))) # FFN
        return x

    def forward(self, x_rgb: torch.Tensor, x_e: torch.Tensor):
        '''
        B C H W, B C H W -> B C H W
        '''
        if self.use_checkpoint:
            return checkpoint.checkpoint(self._forward, x_rgb, x_e)
        else:
            return self._forward(x_rgb, x_e)

class CrossMambaFusionBlock(nn.Module):
    '''
    Cross Mamba Fusion
    '''
    def __init__(
        self,
        hidden_dim: int = 0,
        drop_path: float = 0,
        norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
        attn_drop_rate: float = 0,
        d_state: int = 4,
        dt_rank: Any = "auto",
        ssm_ratio=2.0,
        shared_ssm=False,
        softmax_version=False,
        use_checkpoint: bool = False,
        mlp_ratio=0.0,
        act_layer=nn.GELU,
        drop: float = 0.0,
        **kwargs,
    ):
        super().__init__()
        self.use_checkpoint = use_checkpoint
        # self.norm = norm_layer(hidden_dim)
        self.op = CrossMambaFusion_SS2D_SSM(
            d_model=hidden_dim, 
            dropout=attn_drop_rate, 
            d_state=d_state, 
            ssm_ratio=ssm_ratio, 
            dt_rank=dt_rank,
            shared_ssm=shared_ssm,
            softmax_version=softmax_version,
            **kwargs
        )
        self.drop_path1 = DropPath(drop_path)
        self.drop_path2 = DropPath(drop_path)
        
        self.mlp_branch = mlp_ratio > 0
        if self.mlp_branch:
            self.norm2 = norm_layer(hidden_dim)
            mlp_hidden_dim = int(hidden_dim * mlp_ratio)
            self.mlp = Mlp(in_features=hidden_dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop, channels_first=False)

    def _forward(self, x_rgb: torch.Tensor, x_e: torch.Tensor):
        x_rgb_cross, x_e_cross = self.op(x_rgb, x_e)
        x_rgb = x_rgb + self.drop_path1(x_rgb_cross)
        x_e = x_e + self.drop_path2(x_e_cross)
        return x_rgb, x_e

    def forward(self, x_rgb: torch.Tensor, x_e: torch.Tensor):
        '''
        B C H W, B C H W -> B C H W
        '''
        if self.use_checkpoint:
            return checkpoint.checkpoint(self._forward, x_rgb, x_e)
        else:
            return self._forward(x_rgb, x_e)


class ScaledMambaFusionBlock(nn.Module):
    '''
    scaled mamba fusion, with 2d SSM
    '''
    def __init__(
        self,
        hidden_dim: int = 0,
        drop_path: float = 0,
        norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
        attn_drop_rate: float = 0,
        d_state: int = 4,
        dt_rank: Any = "auto",
        ssm_ratio=2.0,
        shared_ssm=False,
        softmax_version=False,
        use_checkpoint: bool = False,
        mlp_ratio=0.0,
        act_layer=nn.GELU,
        drop: float = 0.0,
        **kwargs,
    ):
        super().__init__()
        self.use_checkpoint = use_checkpoint
        # self.norm = norm_layer(hidden_dim)
        self.op = MM_SS2D(
            d_model=hidden_dim, 
            dropout=attn_drop_rate, 
            d_state=d_state, 
            ssm_ratio=ssm_ratio, 
            dt_rank=dt_rank,
            shared_ssm=shared_ssm,
            softmax_version=softmax_version,
            **kwargs
        )
        self.drop_path = DropPath(drop_path)
        
        self.mlp_branch = mlp_ratio > 0
        if self.mlp_branch:
            self.norm2 = norm_layer(hidden_dim)
            mlp_hidden_dim = int(hidden_dim * mlp_ratio)
            self.mlp = Mlp(in_features=hidden_dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop, channels_first=False)

    def _forward(self, x_rgb: torch.Tensor, x_e: torch.Tensor):
        x = x_rgb + x_e + self.drop_path(self.op(x_rgb, x_e))
        if self.mlp_branch:
            x = x + self.drop_path(self.mlp(self.norm2(x))) # FFN
        return x

    def forward(self, x_rgb: torch.Tensor, x_e: torch.Tensor):
        '''
        B C H W, B C H W -> B C H W
        '''
        if self.use_checkpoint:
            return checkpoint.checkpoint(self._forward, x_rgb, x_e)
        else:
            return self._forward(x_rgb, x_e)

class VSSM(nn.Module):
    def __init__(
            self, 
            patch_size=4, 
            in_chans=3, 
            num_classes=1000, 
            depths=[2, 2, 9, 2], 
            dims=[96, 192, 384, 768], 
            # =========================
            d_state=16, 
            dt_rank="auto", 
            ssm_ratio=2.0, 
            attn_drop_rate=0., 
            shared_ssm=False,
            softmax_version=False,
            # =========================
            drop_rate=0., 
            drop_path_rate=0.1, 
            mlp_ratio=4.0,
            patch_norm=True, 
            norm_layer=nn.LayerNorm,
            downsample_version: str = "v2",
            use_checkpoint=False,  
            **kwargs,
        ):
        super().__init__()
        self.num_classes = num_classes
        self.num_layers = len(depths)
        if isinstance(dims, int):
            dims = [int(dims * 2 ** i_layer) for i_layer in range(self.num_layers)]
        self.embed_dim = dims[0]
        self.num_features = dims[-1]
        self.dims = dims

        self.patch_embed = nn.Sequential(
            nn.Conv2d(in_chans, self.embed_dim, kernel_size=patch_size, stride=patch_size, bias=True),
            Permute(0, 2, 3, 1),
            (norm_layer(self.embed_dim) if patch_norm else nn.Identity()), 
        )

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule

        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):

            if downsample_version == "v2":
                downsample = self._make_downsample(
                    self.dims[i_layer], 
                    self.dims[i_layer + 1], 
                    norm_layer=norm_layer,
                ) if (i_layer < self.num_layers - 1) else nn.Identity()
            else:
                downsample = PatchMerging2D(
                    self.dims[i_layer], 
                    self.dims[i_layer + 1], 
                    norm_layer=norm_layer,
                ) if (i_layer < self.num_layers - 1) else nn.Identity()

            self.layers.append(self._make_layer(
                dim = self.dims[i_layer],
                depth = depths[i_layer],
                drop_path = dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                use_checkpoint=use_checkpoint,
                norm_layer=norm_layer,
                downsample=downsample,
                d_state=d_state,
                dt_rank=dt_rank,
                ssm_ratio=ssm_ratio,
                attn_drop_rate=attn_drop_rate,
                shared_ssm=shared_ssm,
                softmax_version=softmax_version,
                mlp_ratio=mlp_ratio,
                drop_rate=drop_rate,
            ))

        self.classifier = nn.Sequential(OrderedDict(
            norm=norm_layer(self.num_features), # B,H,W,C
            permute=Permute(0, 3, 1, 2),
            avgpool=nn.AdaptiveAvgPool2d(1),
            flatten=nn.Flatten(1),
            head=nn.Linear(self.num_features, num_classes),
        ))

        self.apply(self._init_weights)

    def _init_weights(self, m: nn.Module):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @staticmethod
    def _make_downsample(dim=96, out_dim=192, norm_layer=nn.LayerNorm):
        return nn.Sequential(
            Permute(0, 3, 1, 2),
            nn.Conv2d(dim, out_dim, kernel_size=2, stride=2),
            Permute(0, 2, 3, 1),
            norm_layer(out_dim),
        )

    @staticmethod
    def _make_layer(
        dim=96, 
        depth=2,
        drop_path=[0.1, 0.1], 
        use_checkpoint=False, 
        norm_layer=nn.LayerNorm,
        downsample=nn.Identity(),
        # ===========================
        d_state=16,
        dt_rank="auto",
        ssm_ratio=2.0,
        attn_drop_rate=0.0, 
        shared_ssm=False,
        softmax_version=False,
        # ===========================
        mlp_ratio=4.0,
        drop_rate=0.0,
        **kwargs,
    ):
        assert depth == len(drop_path)
        blocks = []
        for d in range(depth):
            blocks.append(VSSBlock(
                hidden_dim=dim, 
                drop_path=drop_path[d],
                norm_layer=norm_layer,
                attn_drop_rate=attn_drop_rate,
                d_state=d_state,
                dt_rank=dt_rank,
                ssm_ratio=ssm_ratio,
                shared_ssm=shared_ssm,
                softmax_version=softmax_version,
                use_checkpoint=use_checkpoint,
                mlp_ratio=mlp_ratio,
                act_layer=nn.GELU,
                drop=drop_rate,
                **kwargs,
            ))
        
        return nn.Sequential(OrderedDict(
            blocks=nn.Sequential(*blocks,),
            downsample=downsample,
        ))

    def forward(self, x: torch.Tensor):
        x = self.patch_embed(x)
        for layer in self.layers:
            x = layer(x)
        x = self.classifier(x)
        return x

    def flops(self, shape=(3, 224, 224)):
        # shape = self.__input_shape__[1:]
        supported_ops={
            "aten::silu": None, # as relu is in _IGNORED_OPS
            "aten::neg": None, # as relu is in _IGNORED_OPS
            "aten::exp": None, # as relu is in _IGNORED_OPS
            "aten::flip": None, # as permute is in _IGNORED_OPS
            "prim::PythonOp.CrossScan": None,
            "prim::PythonOp.CrossMerge": None,
            "prim::PythonOp.SelectiveScan": selective_scan_flop_jit,
            "prim::PythonOp.SelectiveScanFn": selective_scan_flop_jit,
        }

        model = copy.deepcopy(self)
        model.cuda().eval()

        input = torch.randn((1, *shape), device=next(model.parameters()).device)
        params = parameter_count(model)[""]
        Gflops, unsupported = flop_count(model=model, inputs=(input,), supported_ops=supported_ops)

        del model, input
        return sum(Gflops.values()) * 1e9
        return f"params {params} GFLOPs {sum(Gflops.values())}"

    # used to load ckpt from previous training code
    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):

        def check_name(src, state_dict: dict = state_dict, strict=False):
            if strict:
                if prefix + src in list(state_dict.keys()):
                    return True
            else:
                key = prefix + src
                for k in list(state_dict.keys()):
                    if k.startswith(key):
                        return True
            return False

        def change_name(src, dst, state_dict: dict = state_dict, strict=False):
            if strict:
                if prefix + src in list(state_dict.keys()):
                    state_dict[prefix + dst] = state_dict[prefix + src]
                    state_dict.pop(prefix + src)
            else:
                key = prefix + src
                for k in list(state_dict.keys()):
                    if k.startswith(key):
                        new_k = prefix + dst + k[len(key):]
                        state_dict[new_k] = state_dict[k]
                        state_dict.pop(k)

        change_name("patch_embed.proj", "patch_embed.0")
        change_name("patch_embed.norm", "patch_embed.2")
        for i in range(100):
            for j in range(100):
                change_name(f"layers.{i}.blocks.{j}.ln_1", f"layers.{i}.blocks.{j}.norm")
                change_name(f"layers.{i}.blocks.{j}.self_attention", f"layers.{i}.blocks.{j}.op")
        change_name("norm", "classifier.norm")
        change_name("head", "classifier.head")
        # print('after:', state_dict.keys())

        return super()._load_from_state_dict(state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs)


# compatible with openmmlab
class Backbone_VSSM(VSSM):
    def __init__(self, patch_size=4, in_chans=3, num_classes=1000, 
                 depths=[2, 2, 9, 2], dims=[96, 192, 384, 768], 
                 d_state=16, ssm_ratio=2.0, attn_drop_rate=0., 
                 drop_rate=0., drop_path_rate=0.1, mlp_ratio=4.0,
                 patch_norm=True, norm_layer=nn.LayerNorm,
                 downsample_version: str = "v2",
                 use_checkpoint=False,
                 out_indices=(0, 1, 2, 3), pretrained=None, 
                 **kwargs,
        ):
        super().__init__(patch_size=patch_size, in_chans=in_chans, num_classes=num_classes, 
                         depths=depths, dims=dims, 
                         d_state=d_state, ssm_ratio=ssm_ratio, attn_drop_rate=attn_drop_rate, 
                         drop_rate=drop_rate, drop_path_rate=drop_path_rate, mlp_ratio=mlp_ratio,
                         patch_norm=patch_norm, norm_layer=norm_layer,
                         downsample_version=downsample_version,
                         use_checkpoint=use_checkpoint,
                         **kwargs)
        
        self.out_indices = out_indices
        for i in out_indices:
            layer = norm_layer(self.dims[i])
            layer_name = f'outnorm{i}'
            self.add_module(layer_name, layer)

        del self.classifier
        self.load_pretrained(pretrained)

    def load_pretrained(self, ckpt=None, key="model"):
        if ckpt is None:
            return
        
        try:
            _ckpt = torch.load(open(ckpt, "rb"), map_location=torch.device("cpu"))
            print(f"Successfully load ckpt {ckpt}")
            # print(f"ckpt keys: {_ckpt['model'].keys()}")
            incompatibleKeys = self.load_state_dict(_ckpt[key], strict=False)
            print('incompatible:', incompatibleKeys)        
        except Exception as e:
            print(f"Failed loading checkpoint form {ckpt}: {e}")

    def forward(self, x):
        def layer_forward(l, x):
            x = l.blocks(x)
            y = l.downsample(x)
            return x, y

        x = self.patch_embed(x)
        outs = []
        for i, layer in enumerate(self.layers):
            o, x = layer_forward(layer, x) # (B, H, W, C)
            if i in self.out_indices:
                norm_layer = getattr(self, f'outnorm{i}')
                out = norm_layer(o)
                out = out.permute(0, 3, 1, 2).contiguous()
                outs.append(out)

        if len(self.out_indices) == 0:
            return x
        
        return outs


# ==================================================
def check_vssm_equals_vmambadp():
    try:
        from _ignore.vmamba.vmamba_bak1 import VMamba2Dp
        from _ignore.vmamba.vmamba_pub import VSSM
    except:
        print("original VSSM and VMamba2Dp not found.", flush=True)
        return 

    # test 1 True =================================
    torch.manual_seed(time.time()); torch.cuda.manual_seed(time.time())
    oldvss = VMamba2Dp(depths=[2,2,6,2]).half().cuda()
    newvss = VSSM(depths=[2,2,6,2]).half().cuda()
    newvss.load_state_dict(oldvss.state_dict())
    input = torch.randn((12, 3, 224, 224)).half().cuda()
    torch.cuda.manual_seed(0)
    with torch.cuda.amp.autocast():
        y1 = oldvss.forward_backbone(input)
    torch.cuda.manual_seed(0)
    with torch.cuda.amp.autocast():
        y2 = newvss.forward_backbone(input)
    print((y1 -y2).abs().sum()) # tensor(0., device='cuda:0', grad_fn=<SumBackward0>)
    
    torch.cuda.manual_seed(0)
    with torch.cuda.amp.autocast():
        y1 = oldvss.forward(input)
    torch.cuda.manual_seed(0)
    with torch.cuda.amp.autocast():
        y2 = newvss.forward(input)
    print((y1 -y2).abs().sum()) # tensor(0., device='cuda:0', grad_fn=<SumBackward0>)
    
    # test 2 True ==========================================
    torch.manual_seed(0); torch.cuda.manual_seed(0)
    oldvss = VMamba2Dp(depths=[2,2,6,2]).cuda()
    torch.manual_seed(0); torch.cuda.manual_seed(0)
    newvss = VSSM(depths=[2,2,6,2]).cuda()

    miss_align = 0
    for k, v in oldvss.state_dict().items(): 
        same = (oldvss.state_dict()[k] == newvss.state_dict()[k]).all()
        if not same:
            print(k, same)
            miss_align += 1
    print("init miss align", miss_align) # init miss align 0


def check_vssm1_equals_vssm(ss2dfwd=SS2D.forward_corev0):
    try:
        from _ignore.vmamba.vmamba_pub import VSSM as VSSM0
    except:
        print("original VSSM and VMamba2Dp not found.", flush=True)
        return
    orifwdcore = SS2D.forward_core
    SS2D.forward_core = ss2dfwd

    class VSSM_(VSSM):
        def __init__(
                self, 
                patch_size=4, 
                in_chans=3, 
                num_classes=1000, 
                depths=[2, 2, 9, 2], 
                dims=[96, 192, 384, 768], 
                # =========================
                d_state=16, 
                dt_rank="auto", 
                ssm_ratio=2.0, 
                attn_drop_rate=0., 
                # =========================
                drop_rate=0., 
                drop_path_rate=0.1, 
                mlp_ratio=4.0,
                patch_norm=True, 
                norm_layer=nn.LayerNorm,
                downsample_version: str = "v2",
                use_checkpoint=False,  
                **kwargs,
            ):
            nn.Module.__init__(self)
            self.num_classes = num_classes
            self.num_layers = len(depths)
            if isinstance(dims, int):
                dims = [int(dims * 2 ** i_layer) for i_layer in range(self.num_layers)]
            self.embed_dim = dims[0]
            self.num_features = dims[-1]
            self.dims = dims

            self.patch_embed = nn.Sequential(
                nn.Conv2d(in_chans, self.embed_dim, kernel_size=patch_size, stride=patch_size, bias=True),
                Permute(0, 2, 3, 1),
                (norm_layer(self.embed_dim) if patch_norm else nn.Identity()), 
            )

            dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule

            self.layers = nn.ModuleList()
            for i_layer in range(self.num_layers):

                # if downsample_version == "v2":
                #     downsample = self._make_downsample(
                #         self.dims[i_layer], 
                #         self.dims[i_layer + 1], 
                #         norm_layer=norm_layer,
                #     ) if (i_layer < self.num_layers - 1) else nn.Identity()
                # else:
                #     downsample = PatchMerging2D(
                #         self.dims[i_layer], 
                #         self.dims[i_layer + 1], 
                #         norm_layer=norm_layer,
                #     ) if (i_layer < self.num_layers - 1) else nn.Identity()

                self.layers.append(self._make_layer(
                    dim = self.dims[i_layer],
                    depth = depths[i_layer],
                    drop_path = dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                    use_checkpoint=use_checkpoint,
                    norm_layer=norm_layer,
                    downsample=(i_layer < self.num_layers - 1),
                    d_state=d_state,
                    dt_rank=dt_rank,
                    ssm_ratio=ssm_ratio,
                    attn_drop_rate=attn_drop_rate,
                    mlp_ratio=mlp_ratio,
                    drop_rate=drop_rate,
                ))

            self.classifier = nn.Sequential(OrderedDict(
                norm=norm_layer(self.num_features), # B,H,W,C
                permute=Permute(0, 3, 1, 2),
                avgpool=nn.AdaptiveAvgPool2d(1),
                flatten=nn.Flatten(1),
                head=nn.Linear(self.num_features, num_classes),
            ))
            self.apply(self._init_weights)

        def _make_layer(
            self,
            dim=96, 
            depth=2,
            drop_path=[0.1, 0.1], 
            use_checkpoint=False, 
            norm_layer=nn.LayerNorm,
            downsample=nn.Identity(),
            # ===========================
            d_state=16,
            dt_rank="auto",
            ssm_ratio=2.0,
            attn_drop_rate=0.0, 
            # ===========================
            mlp_ratio=4.0,
            drop_rate=0.0,
            **kwargs,
        ):
            assert depth == len(drop_path)
            blocks = []
            for d in range(depth):
                blocks.append(VSSBlock(
                    hidden_dim=dim, 
                    drop_path=drop_path[d],
                    norm_layer=norm_layer,
                    attn_drop_rate=attn_drop_rate,
                    d_state=d_state,
                    dt_rank=dt_rank,
                    ssm_ratio=ssm_ratio,
                    use_checkpoint=use_checkpoint,
                    mlp_ratio=mlp_ratio,
                    act_layer=nn.GELU,
                    drop=drop_rate,
                    **kwargs,
                ))
                # blocks[d].op = SS2D0(blocks[d].op.d_model)
            

            if True: # is this really applied? Yes, but been overriden later in VSSM!
                def _init_weights(module: nn.Module):
                    for name, p in module.named_parameters():
                        if name in ["out_proj.weight"]:
                            p = p.clone().detach_() # fake init, just to keep the seed ....
                            nn.init.kaiming_uniform_(p, a=math.sqrt(5))
                layer = nn.Sequential(*copy.deepcopy(blocks))
                layer.apply(_init_weights)

            downsample = PatchMerging2D(dim, 2*dim, norm_layer=norm_layer) if downsample else nn.Identity()
            
            return nn.Sequential(OrderedDict(
                blocks=nn.Sequential(*blocks,),
                downsample=downsample,
            ))

        def forward_backbone(self, x):
            x = self.patch_embed(x)
            for l in self.layers:
                x = l(x)
            return x

        def forward1(self, x: torch.Tensor):
            x = self.patch_embed(x)
            for layer in self.layers:
                x = layer(x)
            x = self.classifier.norm(x)
            # here: whether has contiguous would differ
            x = self.classifier.avgpool(x.permute(0, 3, 1, 2).contiguous()).flatten(1)
            x = self.classifier.head(x)
            return x

    VSSM1 = partial(VSSM_, downsample_version="v1", mlp_ratio=0.0, ssm_ratio=2.0, dt_rank="auto", d_state=16)

    # test 1 True =================================
    torch.manual_seed(time.time()); torch.cuda.manual_seed(time.time())
    oldvss = VSSM0(depths=[2,2,6,2]).half().cuda()
    newvss = VSSM1(depths=[2,2,6,2]).half().cuda()
    newvss.load_state_dict(oldvss.state_dict())
    input = torch.randn((12, 3, 224, 224)).half().cuda()
    torch.manual_seed(0); torch.cuda.manual_seed(0)
    with torch.cuda.amp.autocast():
        y1 = oldvss.forward_backbone(input)
    torch.manual_seed(0); torch.cuda.manual_seed(0)
    with torch.cuda.amp.autocast():
        y2 = newvss.forward_backbone(input)
    print((y1 -y2).abs().sum()) # tensor(0., device='cuda:0', grad_fn=<SumBackward0>)
    
    torch.manual_seed(0); torch.cuda.manual_seed(0)
    with torch.cuda.amp.autocast():
        y1 = oldvss.forward(input)
    torch.manual_seed(0); torch.cuda.manual_seed(0)
    with torch.cuda.amp.autocast():
        y2 = newvss.forward1(input)
    print((y1 -y2).abs().sum()) # tensor(0., device='cuda:0', grad_fn=<SumBackward0>)
    torch.manual_seed(0); torch.cuda.manual_seed(0)
    with torch.cuda.amp.autocast():
        y3 = newvss.forward(input)
    print((y1 -y3).abs().sum()) # tensor(0.0008, device='cuda:0', grad_fn=<SumBackward0>)
    
    # test 2 True ==========================================
    torch.manual_seed(0); torch.cuda.manual_seed(0)
    oldvss = VSSM0(depths=[2,2,6,2]).cuda()
    torch.manual_seed(0); torch.cuda.manual_seed(0)
    newvss = VSSM1(depths=[2,2,6,2]).cuda()

    miss_align = 0
    oldvss2new = copy.deepcopy(newvss)
    oldvss2new.load_state_dict(oldvss.state_dict())
    for k, v in oldvss2new.state_dict().items(): 
        same = (oldvss2new.state_dict()[k] == newvss.state_dict()[k]).all()
        if not same:
            print(k, same)
            miss_align += 1
    print("init miss align", miss_align) # init miss align 0
    SS2D.forward_core = orifwdcore


def check_profile():
    vss = VSSM(depths=[1], dims=1024).half().cuda()
    input = torch.randn((128, 3, 56, 56)).half().cuda()
    torch.cuda.manual_seed(0)

    self = vss
    blk = self.layers[0].blocks[0]
    ln_1 = blk.ln_1
    self_attention = blk.self_attention
    selfa = self_attention
    drop_path = blk.drop_path
    input = self.patch_embed(input).detach()

    def trace_handler(prof: torch.profiler.profile):
        print(prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=-1))
        # print(prof.export_chrome_trace("./tracev1.json"))

    with torch.cuda.amp.autocast():
        # with torch.autograd.profiler.profile(enabled=True, use_cuda=True, record_shapes=False, profile_memory=True, with_stack=True) as prof:
        with torch.profiler.profile(
            with_modules=True,
            with_stack=True,
            profile_memory=True,
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],

            # In this example with wait=1, warmup=1, active=2, repeat=1,
            # profiler will skip the first step/iteration,
            # start warming up on the second, record
            # the third and the forth iterations,
            # after which the trace will become available
            # and on_trace_ready (when set) is called;
            # the cycle repeats starting with the next step

            schedule=torch.profiler.schedule(
                wait=1,
                warmup=1,
                active=2,
                repeat=1),
            on_trace_ready=trace_handler
            # on_trace_ready=torch.profiler.tensorboard_trace_handler('./log')
            # used when outputting for tensorboard
            ) as prof:
                for iter in range(1000):
                    x = input
                    # with torch.autograd.profiler.record_function("patch_embed"):
                    #     x = self.patch_embed(x)
                    
                    B, H, W, C = x.shape
                    ori = x

                    with torch.autograd.profiler.record_function("VSSBlock.ln_1"):
                        x = ln_1(x)

                    with torch.autograd.profiler.record_function("SS2D.inproj"):
                        xz = selfa.in_proj(x)
                        x, z = xz.chunk(2, dim=-1) # (b, h, w, d)
                        x = x.permute(0, 3, 1, 2).contiguous()

                    with torch.autograd.profiler.record_function("SS2D.dwconv2d"):
                        x = selfa.act(selfa.conv2d(x)) # (b, d, h, w)
                        # x = self.act(x) # (b, d, h, w)
                    
                    with torch.autograd.profiler.record_function("SS2D.foreward_core"):
                        # y = selfa.forward_corev2(x)
                        # y = selfa.forward_corev3(x)
                        y = selfa.forward_corev1(x)
                        # y = selfa.forward_corev1(x)
                    
                    with torch.autograd.profiler.record_function("SS2D.transpose"):
                        y = torch.transpose(y, dim0=1, dim1=2).contiguous().view(B, H, W, -1)
                        y = selfa.out_norm(y)
                        y = y * F.silu(z)
                    
                    with torch.autograd.profiler.record_function("SS2D.out_proj"):
                        out = selfa.out_proj(y)
                        if selfa.dropout is not None:
                            out = selfa.dropout(out)

                    with torch.autograd.profiler.record_function("SS2D.out"):
                        x = ori + drop_path(out)

                    with torch.autograd.profiler.record_function("backward"):
                        x.sum().backward()

                    prof.step()


def load22kto1k():
    if False:
        # delete relative_position_index since we always re-init it
        relative_position_index_keys = [k for k in state_dict.keys() if "relative_position_index" in k]
        for k in relative_position_index_keys:
            del state_dict[k]

        # delete relative_coords_table since we always re-init it
        relative_position_index_keys = [k for k in state_dict.keys() if "relative_coords_table" in k]
        for k in relative_position_index_keys:
            del state_dict[k]

        # delete attn_mask since we always re-init it
        attn_mask_keys = [k for k in state_dict.keys() if "attn_mask" in k]
        for k in attn_mask_keys:
            del state_dict[k]

        # bicubic interpolate relative_position_bias_table if not match
        relative_position_bias_table_keys = [k for k in state_dict.keys() if "relative_position_bias_table" in k]
        for k in relative_position_bias_table_keys:
            relative_position_bias_table_pretrained = state_dict[k]
            relative_position_bias_table_current = model.state_dict()[k]
            L1, nH1 = relative_position_bias_table_pretrained.size()
            L2, nH2 = relative_position_bias_table_current.size()
            if nH1 != nH2:
                logger.warning(f"Error in loading {k}, passing......")
            else:
                if L1 != L2:
                    # bicubic interpolate relative_position_bias_table if not match
                    S1 = int(L1 ** 0.5)
                    S2 = int(L2 ** 0.5)
                    relative_position_bias_table_pretrained_resized = torch.nn.functional.interpolate(
                        relative_position_bias_table_pretrained.permute(1, 0).view(1, nH1, S1, S1), size=(S2, S2),
                        mode='bicubic')
                    state_dict[k] = relative_position_bias_table_pretrained_resized.view(nH2, L2).permute(1, 0)

        # bicubic interpolate absolute_pos_embed if not match
        absolute_pos_embed_keys = [k for k in state_dict.keys() if "absolute_pos_embed" in k]
        for k in absolute_pos_embed_keys:
            # dpe
            absolute_pos_embed_pretrained = state_dict[k]
            absolute_pos_embed_current = model.state_dict()[k]
            _, L1, C1 = absolute_pos_embed_pretrained.size()
            _, L2, C2 = absolute_pos_embed_current.size()
            if C1 != C1:
                logger.warning(f"Error in loading {k}, passing......")
            else:
                if L1 != L2:
                    S1 = int(L1 ** 0.5)
                    S2 = int(L2 ** 0.5)
                    absolute_pos_embed_pretrained = absolute_pos_embed_pretrained.reshape(-1, S1, S1, C1)
                    absolute_pos_embed_pretrained = absolute_pos_embed_pretrained.permute(0, 3, 1, 2)
                    absolute_pos_embed_pretrained_resized = torch.nn.functional.interpolate(
                        absolute_pos_embed_pretrained, size=(S2, S2), mode='bicubic')
                    absolute_pos_embed_pretrained_resized = absolute_pos_embed_pretrained_resized.permute(0, 2, 3, 1)
                    absolute_pos_embed_pretrained_resized = absolute_pos_embed_pretrained_resized.flatten(1, 2)
                    state_dict[k] = absolute_pos_embed_pretrained_resized

        # check classifier, if not match, then re-init classifier to zero
        head_bias_pretrained = state_dict['head.bias']
        Nc1 = head_bias_pretrained.shape[0]
        Nc2 = model.head.bias.shape[0]
        if (Nc1 != Nc2):
            if Nc1 == 21841 and Nc2 == 1000:
                logger.info("loading ImageNet-22K weight to ImageNet-1K ......")
                map22kto1k_path = f'data/map22kto1k.txt'
                with open(map22kto1k_path) as f:
                    map22kto1k = f.readlines()
                map22kto1k = [int(id22k.strip()) for id22k in map22kto1k]
                state_dict['head.weight'] = state_dict['head.weight'][map22kto1k, :]
                state_dict['head.bias'] = state_dict['head.bias'][map22kto1k]
            else:
                torch.nn.init.constant_(model.head.bias, 0.)
                torch.nn.init.constant_(model.head.weight, 0.)
                del state_dict['head.weight']
                del state_dict['head.bias']
                logger.warning(f"Error in loading classifier head, re-init classifier head to 0")



if __name__ == "__main__":
    check_vssm_equals_vmambadp()
    check_vssm1_equals_vssm(ss2dfwd=SS2D.forward_corev0)
    check_vssm1_equals_vssm(ss2dfwd=SS2D.forward_corev0_seq)
    check_vssm1_equals_vssm(ss2dfwd=SS2D.forward_core)
    check_vssm1_equals_vssm(ss2dfwd=lambda *args, **kwargs: SS2D.forward_corev1(*args, **kwargs).float())

    

