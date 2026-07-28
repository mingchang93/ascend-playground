# 导包（必须先导torch_npu再导torchair）
import torch
import torch_npu
import torchair

# 出现“找不到google或protobuf，或者protobuf版本过高”报错时，需执行如下命令：
# pip3 install protobuf==3.20

# Patch方式实现集合通信入图（可选）
from torchair import patch_for_hcom
patch_for_hcom()

import npu_compat

# 自定义Model
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @torch.compile(dynamic=True)
    def forward(self, x, y):
        return torch.add(x, y)


model = Model().npu()

# 执行编译后的Model
x = torch.randn(2, 2).npu()
y = torch.randn(2, 2).npu()
print(model(x, y))
