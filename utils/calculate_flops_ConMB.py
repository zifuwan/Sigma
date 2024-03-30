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

DEV = False

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
            print(x_rgb_conv.shape, x_e_conv.shape)
            y_rgb, y_e = self.forward_corev2_multimodal(x_rgb_conv, x_e_conv) # b, d, h, w -> b, h, w, d
            # SE to get attention, scale
            b, d, h, w = x_rgb_trans.shape
            x_rgb_squeeze = self.avg_pool(x_rgb_trans).view(b, d)
            x_e_squeeze = self.avg_pool(x_e_trans).view(b, d)
            x_rgb_exitation = self.fc1(x_rgb_squeeze).view(b, d, 1, 1).permute(0, 2, 3, 1).contiguous() # b, 1, 1, d
            x_e_exitation = self.fc2(x_e_squeeze).view(b, d, 1, 1).permute(0, 2, 3, 1).contiguous() 
            y_rgb = y_rgb * x_rgb_exitation
            y_e = y_e * x_e_exitation
            y = torch.concat([y_rgb, y_e], dim=-1)
        out = self.dropout(self.out_proj(y))
        return out

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


class ConMB(nn.Module):
    '''
    Concat Mamba Block, with 2d SSM
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

class MM_SA(nn.Module):
    '''
    Multimodal Mamba Selective Scan 2D
    '''
    def __init__(
        self,
        dim=96,
        num_heads=8,
        qkv_bias=False, 
        qk_scale=None, 
        attn_drop=0., 
        proj_drop=0., 
        sr_ratio=1,
        # ======================
                # basic dims ===========
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
        super().__init__()
        self.dim = dim
            
        if DEV:
            d_conv = -1
            
        factory_kwargs = {"device": None, "dtype": None}
        self.softmax_version = softmax_version
        self.d_model = dim
        self.d_state = math.ceil(self.d_model / 6) if d_state == "auto" else d_state # 20240109
        self.d_conv = d_conv
        self.expand = ssm_ratio
        self.d_inner = int(self.expand * self.d_model)

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
            
        # self attention
        self.num_heads = num_heads
        head_dim = self.d_inner // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        # Linear embedding
        self.q = nn.Linear(self.d_inner, self.d_inner, bias=qkv_bias)
        self.kv = nn.Linear(self.d_inner, self.d_inner * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(self.d_inner, self.d_inner)
        self.proj_drop = nn.Dropout(proj_drop)

        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.sr = nn.Conv2d(self.d_inner, self.d_inner, kernel_size=sr_ratio, stride=sr_ratio)
            self.norm = nn.LayerNorm(self.d_inner)

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

        self.apply(self._init_weights)
        
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()


    def forward(self, x_rgb: torch.Tensor, x_e: torch.Tensor):
        x_rgb = self.in_proj(x_rgb)
        x_e = self.in_proj_modalx(x_e)
        if self.d_conv > 1:
            x_rgb_trans = x_rgb.permute(0, 3, 1, 2).contiguous()
            x_e_trans = x_e.permute(0, 3, 1, 2).contiguous()
            x_rgb_conv = self.act(self.conv2d(x_rgb_trans)) # (b, d, h, w)
            x_e_conv = self.act(self.conv2d_modalx(x_e_trans)) # (b, d, h, w)
            # b, d, h, w -> b, n, d
            b, d, h, w = x_rgb_conv.shape
            x_rgb_conv = x_rgb_conv.flatten(2).transpose(1, 2)
            x_e_conv = x_e_conv.flatten(2).transpose(1, 2)
            # concat in N dim
            x_fuse = torch.cat([x_rgb_conv, x_e_conv], dim=1)
            B, N, C = x_fuse.shape
            # B N C -> B N num_head C//num_head -> B C//num_head N num_heads
            q = self.q(x_fuse).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3) 
            kv = self.kv(x_fuse).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4) 
            k, v = kv[0], kv[1]
            
            attn = (q @ k.transpose(-2, -1)) * self.scale
            attn = F.softmax(attn, dim=-1)
            attn = self.attn_drop(attn)
            x = (attn @ v).transpose(1, 2).reshape(B, N, C)
            x = self.proj(x)
            x = self.proj_drop(x)
            # seperate two modalities
            y_rgb, y_e = torch.split(x, [x_rgb_conv.shape[1], x_e_conv.shape[1],], dim=1)
            # b, n, d -> b, h, w, d
            y_rgb = y_rgb.view(b, h, w, d)
            y_e = y_e.view(b, h, w, d)
            # SE to get attention, scale
            b, d, h, w = x_rgb_trans.shape
            x_rgb_squeeze = self.avg_pool(x_rgb_trans).view(b, d)
            x_e_squeeze = self.avg_pool(x_e_trans).view(b, d)
            x_rgb_exitation = self.fc1(x_rgb_squeeze).view(b, d, 1, 1).permute(0, 2, 3, 1).contiguous() # b, 1, 1, d
            x_e_exitation = self.fc2(x_e_squeeze).view(b, d, 1, 1).permute(0, 2, 3, 1).contiguous() 
            y_rgb = y_rgb * x_rgb_exitation
            y_e = y_e * x_e_exitation
            y = torch.concat([y_rgb, y_e], dim=-1)
        out = self.dropout(self.out_proj(y))
        return out


class ConSA(nn.Module):
    '''
    Concat Self Attention
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
        self.op = MM_SA(dim=hidden_dim)
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

def test_different_stages_sigma():
    models = nn.ModuleList([
        ConMB(hidden_dim=96, mlp_ratio=0.0, d_state=4),
        ConMB(hidden_dim=192, mlp_ratio=0.0, d_state=4),
        ConMB(hidden_dim=384, mlp_ratio=0.0, d_state=4),
        ConMB(hidden_dim=768, mlp_ratio=0.0, d_state=4),
        ])

    x_rgbs = [
        torch.randn(1, 120, 160, 96),
        torch.randn(1, 60, 80, 192),
        torch.randn(1, 30, 40, 384),
        torch.randn(1, 15, 20, 768),
    ]
    x_xs = [
        torch.randn(1, 120, 160, 96),
        torch.randn(1, 60, 80, 192),
        torch.randn(1, 30, 40, 384),
        torch.randn(1, 15, 20, 768),
    ]
    
    # to cuda
    models = models.cuda()
    x_rgbs = [x.cuda() for x in x_rgbs]
    x_xs = [x.cuda() for x in x_xs]
    

    # flops
    flops_fn = partial(selective_scan_flop_jit, nrows=4)
    flops = 0
    for model, x_rgb, x_e in zip(models, x_rgbs, x_xs):
        supported_ops={
            "aten::silu": None, # as relu is in _IGNORED_OPS
            "aten::neg": None, # as relu is in _IGNORED_OPS
            "aten::exp": None, # as relu is in _IGNORED_OPS
            "aten::flip": None, # as permute is in _IGNORED_OPS
            # "prim::PythonOp.CrossScan": None,
            # "prim::PythonOp.CrossMerge": None,
            "prim::PythonOp.SelectiveScanMamba": selective_scan_flop_jit,
            "prim::PythonOp.SelectiveScanOflex": selective_scan_flop_jit,
            "prim::PythonOp.SelectiveScanCore": selective_scan_flop_jit,
            "prim::PythonOp.SelectiveScanNRow": selective_scan_flop_jit,
        }
        Gflops, unsupported = flop_count(model=model, inputs=(x_rgb, x_e), supported_ops=supported_ops)
        print(sum(Gflops.values()), 'Sigma tiny: ---> H: ', x_rgb.shape[1], '---> W: ', x_rgb.shape[2])
        flops += sum(Gflops.values())

def test_different_stages_sa():
    models = nn.ModuleList([
        # ConSA(hidden_dim=96, mlp_ratio=0.0, d_state=4),
        ConSA(hidden_dim=192, mlp_ratio=0.0, d_state=4),
        ConSA(hidden_dim=384, mlp_ratio=0.0, d_state=4),
        ConSA(hidden_dim=768, mlp_ratio=0.0, d_state=4),
        ])

    x_rgbs = [
        # torch.randn(1, 120, 160, 96),
        torch.randn(1, 60, 80, 192),
        torch.randn(1, 30, 40, 384),
        torch.randn(1, 15, 20, 768),
    ]
    x_xs = [
        # torch.randn(1, 120, 160, 96),
        torch.randn(1, 60, 80, 192),
        torch.randn(1, 30, 40, 384),
        torch.randn(1, 15, 20, 768),
    ]
    
    # to cuda
    models = models.cuda()
    models.eval()
    x_rgbs = [x.cuda() for x in x_rgbs]
    x_xs = [x.cuda() for x in x_xs]
    

    # flops
    flops_fn = partial(selective_scan_flop_jit, nrows=4)
    flops = 0
    for model, x_rgb, x_e in zip(models, x_rgbs, x_xs):
        Gflops, unsupported = flop_count(model=model, inputs=(x_rgb, x_e))
        print(sum(Gflops.values()), 'SA: ---> H: ', x_rgb.shape[1], '---> W: ', x_rgb.shape[2])
        flops += sum(Gflops.values())

def test_scaling_sigma():
    model = ConMB(hidden_dim=96, mlp_ratio=0.0, d_state=4)

    x_rgbs = [
        # torch.randn(1, 120, 160, 96),
        # torch.randn(1, 60, 80, 96),
        # torch.randn(1, 60, 60, 96),
        # torch.randn(1, 40, 60, 96),
        # torch.randn(1, 30, 40, 96),
        # torch.randn(1, 15, 20, 96),
        # torch.randn(1, 7, 10, 96),
        # sequence length = 500, 1000, 1500, 2000, 2500
        torch.randn(1, 5, 60, 96),
        torch.randn(1, 10, 60, 96),
        torch.randn(1, 15, 60, 96),
        torch.randn(1, 20, 60, 96),
        torch.randn(1, 25, 60, 96),
    ]
    x_xs = [
        # torch.randn(1, 120, 160, 96),
        # torch.randn(1, 60, 80, 96),
        # torch.randn(1, 60, 60, 96),
        # torch.randn(1, 40, 60, 96),
        # torch.randn(1, 30, 40, 96),
        # torch.randn(1, 15, 20, 96),
        # torch.randn(1, 7, 10, 96),
        torch.randn(1, 5, 60, 96),
        torch.randn(1, 10, 60, 96),
        torch.randn(1, 15, 60, 96),
        torch.randn(1, 20, 60, 96),
        torch.randn(1, 25, 60, 96),
    ]
    
    # to cuda
    model = model.cuda()
    x_rgbs = [x.cuda() for x in x_rgbs]
    x_xs = [x.cuda() for x in x_xs]
    

    # flops
    flops_fn = partial(selective_scan_flop_jit, nrows=4)
    flops = 0
    for x_rgb, x_e in zip(x_rgbs, x_xs):
        supported_ops={
            "aten::silu": None, # as relu is in _IGNORED_OPS
            "aten::neg": None, # as relu is in _IGNORED_OPS
            "aten::exp": None, # as relu is in _IGNORED_OPS
            "aten::flip": None, # as permute is in _IGNORED_OPS
            # "prim::PythonOp.CrossScan": None,
            # "prim::PythonOp.CrossMerge": None,
            "prim::PythonOp.SelectiveScanMamba": selective_scan_flop_jit,
            "prim::PythonOp.SelectiveScanOflex": selective_scan_flop_jit,
            "prim::PythonOp.SelectiveScanCore": selective_scan_flop_jit,
            "prim::PythonOp.SelectiveScanNRow": selective_scan_flop_jit,
        }
        Gflops, unsupported = flop_count(model=model, inputs=(x_rgb, x_e), supported_ops=supported_ops)
        print(sum(Gflops.values()), '---> sequence length: ', x_rgb.shape[1]*x_rgb.shape[2])
        flops += sum(Gflops.values())

def test_scaling_sa():
    model = ConSA(hidden_dim=96, mlp_ratio=0.0, d_state=4)

    x_rgbs = [
        # torch.randn(1, 120, 160, 96),
        # torch.randn(1, 60, 80, 96),
        # torch.randn(1, 60, 60, 96),
        # torch.randn(1, 40, 60, 96),
        # torch.randn(1, 30, 40, 96),
        # torch.randn(1, 15, 20, 96),
        # torch.randn(1, 7, 10, 96),
        # sequence length = 300, 600, 900, 1200, 1500
        torch.randn(1, 5, 60, 96),
        torch.randn(1, 10, 60, 96),
        torch.randn(1, 15, 60, 96),
        torch.randn(1, 20, 60, 96),
        torch.randn(1, 25, 60, 96),
    ]
    x_xs = [
        # torch.randn(1, 120, 160, 96),
        # torch.randn(1, 60, 80, 96),
        # torch.randn(1, 60, 60, 96),
        # torch.randn(1, 40, 60, 96),
        # torch.randn(1, 30, 40, 96),
        # torch.randn(1, 15, 20, 96),
        # torch.randn(1, 7, 10, 96),
        torch.randn(1, 5, 60, 96),
        torch.randn(1, 10, 60, 96),
        torch.randn(1, 15, 60, 96),
        torch.randn(1, 20, 60, 96),
        torch.randn(1, 25, 60, 96),
    ]
    
    # to cuda
    model = model.cuda()
    # x_rgbs = [x.cuda() for x in x_rgbs]
    # x_xs = [x.cuda() for x in x_xs]
    

    # flops
    flops_fn = partial(selective_scan_flop_jit, nrows=4)
    flops = 0
    for x_rgb, x_e in zip(x_rgbs, x_xs):
        
        Gflops, unsupported = flop_count(model=model, inputs=(x_rgb.cuda(), x_e.cuda()))
        print(sum(Gflops.values()), '---> sequence length: ', x_rgb.shape[1]*x_rgb.shape[2])
        flops += sum(Gflops.values())

if __name__ == "__main__":
    # draw teaser figure
    test_scaling_sigma()
    print("=============================================")
    print("=============================================")
    test_scaling_sa()
    
    # compute real flops of different stages
    # test_different_stages_sigma()
    # print("=============================================")
    # print("=============================================")
    # test_different_stages_sa()