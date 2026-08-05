# transfer_to_npu patches torch.cuda.* → torch_npu.npu.* on import
# It does NOT patch torch.compile — flex_attn_patch covers that gap.
import torch
import torch_npu
from torch_npu.contrib import transfer_to_npu  # module-level monkey-patch on import
from torch.nn.attention.flex_attention import create_block_mask, flex_attention
import npu_compat

device = "npu"
torch.set_default_device(device)

B, H, Q_LEN, KV_LEN = 2, 4, 1024, 1024
BLOCK_SIZE = 128


def causal_mask(b, h, q_idx, kv_idx):
    return q_idx >= kv_idx


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        block_mask = create_block_mask(
            causal_mask, B=B, H=H, Q_LEN=Q_LEN, KV_LEN=KV_LEN,
            BLOCK_SIZE=BLOCK_SIZE, device="npu",
        )
        self.block_mask = block_mask

    def forward(self, query, key, value):
        return flex_attention(query, key, value, block_mask=self.block_mask)


model = Model()

head_dim = 64
query = torch.randn(B, H, Q_LEN, head_dim, device=device, dtype=torch.float16)
key = torch.randn(B, H, KV_LEN, head_dim, device=device, dtype=torch.float16)
value = torch.randn(B, H, KV_LEN, head_dim, device=device, dtype=torch.float16)
output = model(query, key, value)
print(output.shape)
