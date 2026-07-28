import torch
from torch_compile_patch.compile_util import compile_model

# 自定义Model
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, y):
        return torch.add(x, y)

device = "npu" if torch.npu.is_available() else "cuda" if torch.cuda.is_available() else "cpu"

model = Model().to(device)
model = compile_model(model, device=device, dynamic=True)

# 执行编译后的Model
x = torch.randn(2, 2).to(device)
y = torch.randn(2, 2).to(device)
print(model(x, y))
