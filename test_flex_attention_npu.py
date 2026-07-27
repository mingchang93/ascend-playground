import torch


# patch BEFORE importing flex_attention
import patch_flex_attention_npu

from torch.nn.attention import flex_attention as fa_mod
from torch.nn.attention.flex_attention import flex_attention

DEVICE = "npu"


def make_qkv(batch=1, heads=8, seq=128, dim=64):
    """Create test QKV tensors. Avoids randn_like internal-format warning."""
    shape = (batch, heads, seq, dim)
    q = torch.randn(shape, device=DEVICE, dtype=torch.float16)
    k = torch.randn(shape, device=DEVICE, dtype=torch.float16)
    v = torch.randn(shape, device=DEVICE, dtype=torch.float16)
    return q, k, v


# ── Test 1: patch status ──────────────────────────────────────────

def test_patch():
    print("\n====== TEST 1: PATCH STATUS ======")

    assert getattr(fa_mod, "_npu_patch_applied", False), "Patch not installed"
    print("  Patch installed: OK")

    assert callable(fa_mod._validate_device), "Validate not callable"
    print("  _validate_device replaced: OK")

    # Check if compile-debug flag exists (added in PyTorch 2.8+)
    if hasattr(fa_mod, "_FLEX_ATTENTION_DISABLE_COMPILE_DEBUG"):
        print(f"  _FLEX_ATTENTION_DISABLE_COMPILE_DEBUG = "
              f"{fa_mod._FLEX_ATTENTION_DISABLE_COMPILE_DEBUG}")
    else:
        print("  _FLEX_ATTENTION_DISABLE_COMPILE_DEBUG: n/a (PyTorch < 2.8)")


# ── Test 2: device validation bypass ──────────────────────────────

def test_device_validation():
    print("\n====== TEST 2: DEVICE VALIDATION ======")
    q, k, v = make_qkv()

    try:
        fa_mod._validate_device(q, k, v)
        print("  NPU validation bypass: PASSED")
    except Exception as e:
        print(f"  NPU validation bypass: FAILED — {type(e).__name__}: {e}")
        raise


# ── Test 3: eager execution (no compile) ──────────────────────────

def test_flex_attention_eager():
    print("\n====== TEST 3: FLEX ATTENTION EAGER ======")
    q, k, v = make_qkv()

    try:
        out = flex_attention(q, k, v)
        print(f"  Output shape:  {out.shape}")
        print(f"  Output device: {out.device}")
        print("  Eager execution: PASSED")
        return True
    except Exception as e:
        print(f"  Eager execution: FAILED")
        print(f"  {type(e).__name__}: {e}")
        return False


# ── Test 4: torch.compile across backends ─────────────────────────

def test_compile_backends():
    """Try each compile backend to find which (if any) works on NPU."""
    print("\n====== TEST 4: TORCH.COMPILE BACKENDS ======")

    # Backends ordered by likelihood on NPU:
    #   "inductor"     — full Triton codegen; needs triton-ascend working
    #   "aot_eager"    — AOTAutograd graph capture, no Triton codegen
    #   "eager"        — fx graph capture only, no codegen at all
    backends = ["inductor", "aot_eager", "eager"]

    q, k, v = make_qkv()

    def fn(q, k, v):
        return flex_attention(q, k, v)

    results = {}
    for backend in backends:
        try:
            torch.compile(fn, backend=backend)(q, k, v)
            results[backend] = "PASS"
        except Exception as e:
            # Print full error for inductor (the one that matters)
            msg = str(e)
            if backend == "inductor":
                # Show the last meaningful line: the root cause
                lines = [l for l in msg.split("\n") if l.strip()]
                results[backend] = f"FAIL — {lines[-1].strip() if lines else type(e).__name__}"
            else:
                results[backend] = f"FAIL — {type(e).__name__}: {msg.split(chr(10))[0]}"

    for backend, result in results.items():
        print(f"  {backend:12s}: {result}")

    working = [b for b, r in results.items() if "PASS" in r]
    inductor_ok = "PASS" in results.get("inductor", "")

    if inductor_ok:
        print("  → Inductor works: full Triton codegen on NPU")
    elif working:
        print(f"  → Inductor FAILED. Fallback: torch.compile(fn, backend=\"{working[0]}\")")
        print("    (aot_eager/eager = graph capture only, no Triton kernel generation)")
    else:
        print("  → No compile backend works. Use eager mode.")

    return results


# ── Main ──────────────────────────────────────────────────────────

def main():
    print(f"PyTorch: {torch.__version__}")

    if not torch.npu.is_available():
        raise RuntimeError("NPU not available — this test requires an Ascend NPU device")

    test_patch()
    test_device_validation()

    eager_ok = test_flex_attention_eager()
    compile_results = test_compile_backends()

    # ── Summary ──
    print("\n========== SUMMARY ==========")
    print(f"  Eager (no compile):  {'PASS' if eager_ok else 'FAIL'}")

    inductor_ok = "PASS" in compile_results.get("inductor", "")
    working = [b for b, r in compile_results.items() if "PASS" in r]

    if inductor_ok:
        print(f"  torch.compile:       PASS — Inductor Triton codegen on NPU")
    elif working:
        print(f"  torch.compile:       FALLBACK — inductor failed, {working[0]} works (graph capture only)")
    else:
        print(f"  torch.compile:       FAIL — all backends")

    if not eager_ok:
        print("\n  → FlexAttention kernel missing for NPU (PrivateUse1).")
        print("    Need: NPU kernel or fallback to SDPA / torch_npu.npu_fusion_attention.")
    elif inductor_ok:
        print("\n  → Full FlexAttention + Inductor Triton compilation works on NPU.")
    elif working:
        print("\n  → Eager kernel works. Inductor Triton codegen fails.")
        print("    Check the inductor debug trace for the root cause.")
    else:
        print("\n  → Kernel exists (eager works) but no compile backend works on NPU.")

    print("============================\n")


if __name__ == "__main__":
    main()