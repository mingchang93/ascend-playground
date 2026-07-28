# 导包（必须先导torch_npu再导torchair）
import torch
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
