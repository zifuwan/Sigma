import numpy as np
import torch.nn as nn
import torch

from torch.nn.modules import module
import torch.nn.functional as F
from models.encoders.vmamba import CVSSDecoderBlock
import torch.utils.checkpoint as checkpoint
from einops import rearrange


class PatchExpand(nn.Module):
    def __init__(self, input_resolution, dim, dim_scale=2, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.expand = nn.Linear(dim, 2 * dim, bias=False) if dim_scale == 2 else nn.Identity()
        self.norm = norm_layer(dim // dim_scale)

    def forward(self, x):
        """
        x: B, H, W, C
        """
        
        x = self.expand(x) # B, H, W, 2C
        B, H, W, C = x.shape
        x = rearrange(x, 'b h w (p1 p2 c)-> b (h p1) (w p2) c', p1=2, p2=2, c=C // 4)
        x = self.norm(x)

        return x


class UpsampleExpand(nn.Module):
    def __init__(self, input_resolution, dim, patch_size=4, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.patch_size = patch_size
        self.linear = nn.Linear(dim, dim // 2, bias=False)
        self.output_dim = dim
        self.norm = norm_layer(dim // 2)
    
    def forward(self, x):
        """
        x: B, H, W, C
        """
        B, H, W, C = x.shape
        x = self.linear(x).permute(0, 3, 1, 2).contiguous() # B, C/2, H, W
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False).permute(0, 2, 3, 1).contiguous() # B, 2H, 2W, C/2
        x = self.norm(x)
        return x
        

class FinalPatchExpand_X4(nn.Module):
    def __init__(self, input_resolution, dim, patch_size=4, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.patch_size = patch_size
        self.expand = nn.Linear(dim, patch_size * patch_size * dim, bias=False)
        self.output_dim = dim
        self.norm = norm_layer(self.output_dim)

    def forward(self, x):
        """
        x: B, H, W, C
        """
        x = self.expand(x) # B, H, W, 16C
        B, H, W, C = x.shape
        x = rearrange(x, 'b h w (p1 p2 c)-> b (h p1) (w p2) c', p1=self.patch_size, p2=self.patch_size,
                      c=C // (self.patch_size ** 2))
        x = self.norm(x)
        return x


class FinalUpsample_X4(nn.Module):
    def __init__(self, input_resolution, dim, patch_size=4, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.patch_size = patch_size
        self.linear1 = nn.Linear(dim, dim, bias=False)
        self.linear2 = nn.Linear(dim, dim, bias=False)
        self.output_dim = dim
        self.norm = norm_layer(self.output_dim)
    
    def forward(self, x):
        """
        x: B, H, W, C
        """
        B, H, W, C = x.shape
        x = self.linear1(x).permute(0, 3, 1, 2).contiguous() # B, C, H, W
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False).permute(0, 2, 3, 1).contiguous() # B, 2H, 2W, C
        x = self.linear2(x).permute(0, 3, 1, 2).contiguous() # B, C, 2H, 2W
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False).permute(0, 2, 3, 1).contiguous() # B, 4H, 4W, C
        x = self.norm(x)
        return x
        


class Mamba_up(nn.Module):
    def __init__(self, dim, input_resolution, depth, dt_rank="auto", 
                 d_state=4, ssm_ratio=2.0, attn_drop_rate=0., 
                 drop_rate=0.0, mlp_ratio=4.0,
                 drop_path=0.1, norm_layer=nn.LayerNorm, upsample=None, 
                 shared_ssm=False, softmax_version=False,
                 use_checkpoint=False, **kwargs):

        super().__init__()
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint

        # build blocks
        self.blocks = nn.ModuleList([
            CVSSDecoderBlock(
                hidden_dim=dim, 
                drop_path=drop_path[i],
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
            )
            for i in range(depth)])

        # patch merging layer
        if upsample is not None:
            # self.upsample = PatchExpand(input_resolution, dim=dim, dim_scale=2, norm_layer=norm_layer)
            self.upsample = UpsampleExpand(input_resolution, dim=dim, patch_size=2, norm_layer=norm_layer)
        else:
            self.upsample = None

    def forward(self, x):
        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x)
            else:
                x = blk(x)
        if self.upsample is not None:
            x = self.upsample(x)
        return x


class MambaDecoder(nn.Module):
    def __init__(self, 
                 img_size=[480, 640],
                 in_channels=[96, 192, 384, 768], #[64, 128, 320, 512],
                 num_classes=40,
                 dropout_ratio=0.1,
                 embed_dim=96,
                 align_corners=False,
                 patch_size=4, 
                 depths=[4, 4, 4, 4], 
                 mlp_ratio=4.,
                 drop_rate=0.0,
                 attn_drop_rate=0.,
                 drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm,
                 use_checkpoint=False,
                 deep_supervision=False,
                 **kwargs):
        super().__init__()

        self.num_classes = num_classes
        self.num_layers = len(depths) # actually only three depths are used. The last feature is simply upexpanded
        self.mlp_ratio = mlp_ratio
        self.patch_size = patch_size
        self.patches_resolution = [img_size[0] // patch_size, img_size[1] // patch_size]
        self.deep_supervision = deep_supervision

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

        self.layers_up = nn.ModuleList()
        for i_layer in range(self.num_layers):
            if i_layer == 0:
                # B, 768, 15, 20 -> B, 384, 30, 40
                layer_up = PatchExpand(
                    input_resolution=(self.patches_resolution[0] // (2 ** (self.num_layers - 1 - i_layer)),
                                      self.patches_resolution[1] // (2 ** (self.num_layers - 1 - i_layer))),
                    dim=int(embed_dim * 2 ** (self.num_layers - 1 - i_layer)), 
                    dim_scale=2, 
                    norm_layer=norm_layer)
            else:
                # B, 30, 40, 384 -> B, 60, 80, 192
                # B, 60, 80, 192 -> B, 120, 160, 96
                # B, 120, 160, 96 -> B, 120, 160, 96
                layer_up = Mamba_up(dim=int(embed_dim * 2 ** (self.num_layers - 1 - i_layer)),
                                         input_resolution=(
                                         self.patches_resolution[0] // (2 ** (self.num_layers - 1 - i_layer)),
                                         self.patches_resolution[1] // (2 ** (self.num_layers - 1 - i_layer))),
                                         depth=depths[(self.num_layers - 1 - i_layer)],
                                         mlp_ratio=self.mlp_ratio,
                                         drop=drop_rate, 
                                         attn_drop=attn_drop_rate,
                                         drop_path=dpr[sum(depths[:(self.num_layers - 1 - i_layer)]):sum(
                                             depths[:(self.num_layers - 1 - i_layer) + 1])],
                                         norm_layer=norm_layer,
                                         upsample=PatchExpand if (i_layer < self.num_layers - 1) else None,
                                         use_checkpoint=use_checkpoint)
            self.layers_up.append(layer_up)

        self.norm_up = norm_layer(embed_dim)
        if self.deep_supervision:
            self.norm_ds = nn.ModuleList([norm_layer(embed_dim * 2 ** (self.num_layers - 2 - i_layer)) for i_layer in range(self.num_layers-1)])
            self.output_ds = nn.ModuleList([nn.Conv2d(in_channels=embed_dim * 2 ** (self.num_layers - 2 - i_layer), out_channels=self.num_classes, kernel_size=1, bias=False) for i_layer in range(self.num_layers-1)])
        
        # print("---final upsample expand_first---")
        # self.up = FinalPatchExpand_X4(input_resolution=(img_size[0] // patch_size, img_size[1] // patch_size),
        #                                 patch_size=4, dim=embed_dim)
        self.up = FinalUpsample_X4(input_resolution=(img_size[0] // patch_size, img_size[1] // patch_size),
                                      patch_size=4, dim=embed_dim)
        self.output = nn.Conv2d(in_channels=embed_dim, out_channels=self.num_classes, kernel_size=1, bias=False)

        
    def forward_up_features(self, inputs):  # B, C, H, W
        if not self.deep_supervision:
            for inx, layer_up in enumerate(self.layers_up):
                if inx == 0:
                    x = inputs[3 - inx]  # B, 768, 15, 20
                    x = x.permute(0, 2, 3, 1).contiguous()  # B, 15, 20, 768
                    y = layer_up(x)  # B, 30, 40, 384
                else:
                    # interpolate y to input size (only pst900 dataset needs)
                    B, C, H, W = inputs[3 - inx].shape
                    y = F.interpolate(y.permute(0, 3, 1, 2).contiguous(), size=(H, W), mode='bilinear', align_corners=False).permute(0, 2, 3, 1).contiguous()
                    
                    x = y + inputs[3 - inx].permute(0, 2, 3, 1).contiguous()
                    y = layer_up(x)

            x = self.norm_up(y)

            return x
        else:
            # if deep supervision
            x_upsample = []
            for inx, layer_up in enumerate(self.layers_up):
                if inx == 0:
                    x = inputs[3 - inx]  # B, 768, 15, 20
                    x = x.permute(0, 2, 3, 1).contiguous()  # B, 15, 20, 768
                    y = layer_up(x)  # B, 30, 40, 384
                    x_upsample.append(self.norm_ds[inx](y))
                else:
                    x = y + inputs[3 - inx].permute(0, 2, 3, 1).contiguous()
                    y = layer_up(x)
                    if inx != self.num_layers - 1:
                        x_upsample.append((self.norm_ds[inx](y)))

            x = self.norm_up(y)

            return x, x_upsample
    
    def forward(self, inputs):
        if not self.deep_supervision:
            x = self.forward_up_features(inputs) # B, H, W, C
            x_last = self.up_x4(x, self.patch_size)
            return x_last
        else:
            x, x_upsample = self.forward_up_features(inputs)
            x_last = self.up_x4(x, self.patch_size)
            x_output_0 = self.output_ds[0](F.interpolate(x_upsample[0].permute(0,3,1,2).contiguous(), scale_factor=16, mode='bilinear', align_corners=False))
            x_output_1  = self.output_ds[1](F.interpolate(x_upsample[1].permute(0,3,1,2).contiguous(), scale_factor=8, mode='bilinear', align_corners=False))
            x_output_2 = self.output_ds[2](F.interpolate(x_upsample[2].permute(0,3,1,2).contiguous(), scale_factor=4, mode='bilinear', align_corners=False))
            return x_last, x_output_0, x_output_1, x_output_2
    
    def up_x4(self, x, pz):
        B, H, W, C = x.shape

        x = self.up(x)
        x = x.view(B, pz * H, pz * W, -1)
        x = x.permute(0, 3, 1, 2).contiguous()  # B, C, 4H, 4W
        x = self.output(x)

        return x