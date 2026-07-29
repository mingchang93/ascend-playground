# transfer_to_npu patches torch.cuda.* → torch_npu.npu.* on import
# It does NOT patch torch.compile — flex_attn_patch covers that gap.
import torch
import torch_npu
from torch_npu.contrib import transfer_to_npu  # module-level monkey-patch on import
import npu_compat

class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @torch.compile(dynamic=True)
    def forward(self, x, y):
        return torch.add(x, y)


model = Model()  # transfer_to_npu patches .cuda → .npu

# transfer_to_npu patches torch.randn etc. to accept device="npu"
x = torch.randn(2, 2)
y = torch.randn(2, 2)

print(model(x, y))
