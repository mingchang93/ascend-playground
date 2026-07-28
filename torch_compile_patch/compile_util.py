# compile_utils.py — backend selection lives here, once
import torch

def compile_model(model, device: str, dynamic: bool = True, enabled: bool = True):
    if not enabled:
        return model

    if device == "npu":
        import torch_npu
        import torchair
        config = torchair.CompilerConfig()
        backend = torchair.get_npu_backend(compiler_config=config)
        return torch.compile(model, backend=backend, dynamic=dynamic)

    # GPU / CPU: default inductor backend
    return torch.compile(model, dynamic=dynamic)
