# 导包（必须先导torch_npu再导torchair）
import torch
import torch_npu
import torchair

# Patch方式实现集合通信入图（可选）
from torchair import patch_for_hcom
patch_for_hcom()

# 自定义Model
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x, y):
        return torch.add(x, y)
model = Model().npu()

# 配置图模式config
config = torchair.CompilerConfig()
# 配置图执行模式，默认max-autotune
# config.mode = "reduce-overhead"
npu_backend = torchair.get_npu_backend(compiler_config=config)
# 基于NPU backend进行compile
opt_model = torch.compile(model, backend=npu_backend)

# 执行编译后的Model
x = torch.randn(2, 2).npu()
y = torch.randn(2, 2).npu()
print(opt_model(x, y))
