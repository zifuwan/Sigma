import torch
from fvcore.nn import FlopCountAnalysis, parameter_count_table
from ..models.builder import EncoderDecoder as segmodel
from config import config
import torch.nn as nn
from fvcore.nn import FlopCountAnalysis, parameter_count_table


network = segmodel(cfg=config, criterion=None, norm_layer=nn.BatchNorm2d).cuda(4)

network.eval()
# init input
tensor = (torch.randn(1, 3, 480, 640).cuda(4),torch.randn(1, 3, 480, 640).cuda(4))

# claculate FLOPs
flops = FlopCountAnalysis(network, tensor)

print("FLOPs: ", flops.total()/1e9,  'G')
print(parameter_count_table(network))

# calculate with code from https://github.com/MzeroMiko/VMamba/blob/main/classification/models/vmamba.py#L4
# no difference found.
# n_parameters = sum(p.numel() for p in network.parameters() if p.requires_grad)
# print(f"number of params: {n_parameters}")
# flops = network.flops()
# print(f"number of GFLOPs: {flops / 1e9}")