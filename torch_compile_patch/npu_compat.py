# npu_compat.py — import this once, before any model code
import torch

def _make_compile_shim():
    _original_compile = torch.compile

    try:
        import torch_npu
        import torchair
        _npu_available = torch.npu.is_available()
    except ImportError:
        _npu_available = False

    if _npu_available:
        _config = torchair.CompilerConfig()
        _npu_backend = torchair.get_npu_backend(compiler_config=_config)

    def compile_shim(model=None, *, backend=None, dynamic=None, **kwargs):
        if _npu_available and backend is None:
            backend = _npu_backend  # only override if caller didn't specify one
        return _original_compile(model, backend=backend, dynamic=dynamic, **kwargs)

    return compile_shim

if not getattr(torch.compile, "_is_npu_shim", False):
    torch.compile = _make_compile_shim()
    torch.compile._is_npu_shim = True
