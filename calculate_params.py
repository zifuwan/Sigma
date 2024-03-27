import torch
from fvcore.nn import FlopCountAnalysis, parameter_count_table
from models.builder import EncoderDecoder as segmodel
from config import config
import torch.nn as nn
from fvcore.nn import FlopCountAnalysis, parameter_count_table


network = segmodel(cfg=config, criterion=None, norm_layer=nn.BatchNorm2d).cuda(4)

network.eval()
# 创建输入网络的tensor
tensor = (torch.randn(1, 3, 480, 640).cuda(4),torch.randn(1, 3, 480, 640).cuda(4))

# 分析FLOPs
flops = FlopCountAnalysis(network, tensor)
# print("FLOPs: ", flops.total())
# 以G为单位
print("FLOPs: ", flops.total()/1e9,  'G')
print(parameter_count_table(network))