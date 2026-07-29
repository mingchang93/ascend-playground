# transfer_to_npu patches torch.cuda.* → torch_npu.npu.* on import
# It does NOT patch torch.compile — flex_attn_patch covers that gap.
import torch
import torch_npu
from torch_npu.contrib import transfer_to_npu  # module-level monkey-patch on import

# Apply flex_attention NPU compile patch (covers torch.compile gap)
import flex_attn_patch.patch_flex_attention_npu  # noqa: F401


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, y):
        return torch.add(x, y)


model = Model()  # transfer_to_npu patches .cuda → .npu

# transfer_to_npu patches torch.randn etc. to accept device="npu"
x = torch.randn(2, 2)
y = torch.randn(2, 2)

# torch.compile with NPU backend (from flex_attn_patch)
compiled = torch.compile(model, dynamic=True)
print(compiled(x, y))
