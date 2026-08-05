# FlexAttention NPU Patch

Monkey-patch PyTorch FlexAttention's device validation to accept Ascend NPU tensors, and optionally enable Triton-based compilation.

## Quick Start

### Bare Metal (conda)

```bash
# 1. Create conda environment
conda create -n flexattn-npu python=3.11 -y
conda activate flexattn-npu

# 2. Install TorchNPU
pip install torch-npu==2.7.1.post4

# 3. (Optional) Install Triton for Ascend — enables torch.compile
pip install triton-ascend==3.2.1 \
  --extra-index-url=https://mirrors.huaweicloud.com/ascend/repos/pypi

# 4. Run the test
python test_flex_attention_npu.py
```

### Docker (pre-built image)

The image includes CANN, PyTorch, and TorchNPU. Only triton-ascend is extra.

```bash
docker run -u 0 -dit --shm-size=512g --name=flexattn-npu \
  --net=host --privileged \
  --security-opt seccomp=unconfined \
  --device=/dev/davinci0 \
  --device=/dev/davinci1 \
  --device=/dev/davinci2 \
  --device=/dev/davinci3 \
  --device=/dev/davinci4 \
  --device=/dev/davinci5 \
  --device=/dev/davinci6 \
  --device=/dev/davinci7 \
  --device=/dev/davinci_manager \
  --device=/dev/devmm_svm \
  --device=/dev/hisi_hdc \
  -v /usr/local/dcmi:/usr/local/dcmi \
  -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \
  -v /usr/local/sbin/npu-smi:/usr/local/sbin/npu-smi \
  -v /usr/local/Ascend/driver:/usr/local/Ascend/driver \
  -v /etc/ascend_install.info:/etc/ascend_install.info \
  -v /data1:/data1 \
  swr.cn-south-1.myhuaweicloud.com/ascendhub/torch-npu:2.7.1.post4-910b-ubuntu22.04-py3.11 \
  /bin/bash



docker run -u 0 -dit --shm-size=512g --name=flexattn-npu \
  --net=host --privileged \
  --security-opt seccomp=unconfined \
  --device=/dev/davinci0 \
  --device=/dev/davinci1 \
  --device=/dev/davinci2 \
  --device=/dev/davinci3 \
  --device=/dev/davinci4 \
  --device=/dev/davinci5 \
  --device=/dev/davinci6 \
  --device=/dev/davinci7 \
  --device=/dev/davinci8 \
  --device=/dev/davinci9 \
  --device=/dev/davinci10 \
  --device=/dev/davinci11 \
  --device=/dev/davinci12 \
  --device=/dev/davinci13 \
  --device=/dev/davinci14 \
  --device=/dev/davinci15 \
  --device=/dev/davinci_manager \
  --device=/dev/devmm_svm \
  --device=/dev/hisi_hdc \
  -v /usr/local/dcmi:/usr/local/dcmi \
  -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \
  -v /usr/local/sbin/npu-smi:/usr/local/sbin/npu-smi \
  -v /usr/local/Ascend/driver:/usr/local/Ascend/driver \
  -v /etc/ascend_install.info:/etc/ascend_install.info \
  -v /data1:/data1 \
  swr.cn-south-1.myhuaweicloud.com/ascendhub/torch-npu:2.7.1.post4-a3-ubuntu22.04-py3.11 \
  /bin/bash


# Inside container:
docker exec -it flexattn-npu bash
pip install triton-ascend==3.2.1 \
  --extra-index-url=https://mirrors.huaweicloud.com/ascend/repos/pypi
python test_flex_attention_npu.py
```

---

## Prerequisites

| Component | Required | Recommended |
|-----------|----------|-------------|
| NPU | Ascend 910B / 910C / Atlas A2/A3 | 910B |
| CANN | ≥ 8.5.0 | 9.0.0 |
| Python | 3.9 – 3.11 | 3.11 |
| OS | Linux (aarch64 / x86_64) | Ubuntu 22.04 |
| Memory | ≥ 32 GB | — |

---

## Detailed Setup

### 1. CANN Toolkit

Check your version:

```bash
cat /usr/local/Ascend/ascend-toolkit/latest/version.cfg
```

If CANN < 9.0.0, download from [HiAscend](https://www.hiascend.com/developer/download/community/result?module=cann) and install:

```bash
./Ascend-cann-toolkit_9.0.0_linux-$(uname -m).run --install
```

### 2. Conda Environment

```bash
conda create -n flexattn-npu python=3.11 -y
conda activate flexattn-npu
```

### 3. TorchNPU

TorchNPU is the PyTorch adapter for Ascend NPU. It provides the `npu` device backend.

```bash
pip install torch-npu==2.7.1.post4
```

Verify:

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh

python -c "
import torch
import torch_npu
print('PyTorch:', torch.__version__)
print('NPU available:', torch.npu.is_available())
x = torch.randn(2, 2).npu()
print('NPU tensor:', x.device)
"
```

### 4. Triton-Ascend (optional — for torch.compile)

[triton-lang/triton-ascend](https://github.com/triton-lang/triton-ascend) is the official Triton compiler port for Ascend NPU. It enables `torch.compile(backend="inductor")` on NPU.

```bash
pip install triton-ascend==3.2.1 \
  --extra-index-url=https://mirrors.huaweicloud.com/ascend/repos/pypi
```

Verify:

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh

python -c "
import triton
print('Triton:', triton.__version__)

import triton.language as tl
import torch

@triton.jit
def add_kernel(x_ptr, y_ptr, out_ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask)
    y = tl.load(y_ptr + offs, mask=mask)
    tl.store(out_ptr + offs, x + y, mask=mask)

n = 1024
x = torch.randn(n, device='npu', dtype=torch.float16)
y = torch.randn(n, device='npu', dtype=torch.float16)
out = torch.empty_like(x)
add_kernel[(n // 256 + 1,)](x, y, out, n, BLOCK=256)
print('Triton kernel OK:', torch.allclose(out, x + y, atol=1e-2))
"
```

---

## Files

| File | Purpose |
|------|---------|
| `patch_flex_attention_npu.py` | Monkey-patches `_validate_device` to accept NPU tensors |
| `test_flex_attention_npu.py` | 4-layer test: patch status → validation → eager → compile backends |

---

## Test Output Interpretation

| Eager | Compile | Diagnosis |
|-------|---------|-----------|
| PASS | PASS | Full FlexAttention + compile works on NPU |
| PASS | FAIL | Kernel exists, compile path broken. Install triton-ascend. |
| FAIL | FAIL | FlexAttention kernel missing for NPU. Fall back to SDPA or `torch_npu.npu_fusion_attention`. |

---

## How It Works

PyTorch's `flex_attention` calls `_validate_device(query, key, value)` which rejects non-CUDA tensors. The patch wraps it:

```python
def npu_validate(query, key, value):
    if query.device.type == "npu":
        return                        # bypass
    return original_validate(query, key, value)
```

This is a **device guard bypass only** — it does not implement the NPU kernel. The actual kernel comes from TorchNPU (eager path) or Triton-Ascend (compile path).