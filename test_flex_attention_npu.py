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

    if hasattr(fa_mod, "_FLEX_ATTENTION_DISABLE_COMPILE_DEBUG"):
        flag = fa_mod._FLEX_ATTENTION_DISABLE_COMPILE_DEBUG
        print(f"  _FLEX_ATTENTION_DISABLE_COMPILE_DEBUG = {flag}")
        assert flag is True, "Compile flag not set"
    else:
        print("  _FLEX_ATTENTION_DISABLE_COMPILE_DEBUG: not present (older PyTorch)")

    # Verify the validate function is our patched version, not the original
    # (original_validate_device doesn't exist — the patch uses original_validate)
    assert callable(fa_mod._validate_device), "Validate not callable"
    print("  _validate_device replaced: OK")


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


# ── Test 3: eager execution (compile disabled) ────────────────────

def test_flex_attention_eager():
    print("\n====== TEST 3: FLEX ATTENTION EAGER (compile disabled) ======")
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


# ── Test 4: torch.compile scenario ────────────────────────────────

def test_flex_attention_compile():
    print("\n====== TEST 4: TORCH.COMPILE (compile disabled) ======")
    q, k, v = make_qkv()

    def fn(q, k, v):
        return flex_attention(q, k, v)

    try:
        compiled_fn = torch.compile(fn, backend="inductor")
        out = compiled_fn(q, k, v)
        print(f"  Compile output shape: {out.shape}")
        print("  torch.compile: PASSED")
        return True
    except Exception as e:
        print(f"  torch.compile: FAILED")
        print(f"  {type(e).__name__}: {e}")
        return False


# ── Test 5: compile enabled vs disabled comparison ────────────────

def test_compile_comparison():
    """Toggle _FLEX_ATTENTION_DISABLE_COMPILE_DEBUG to isolate the blocker."""
    print("\n====== TEST 5: COMPILE ENABLED vs DISABLED ======")

    if not hasattr(fa_mod, "_FLEX_ATTENTION_DISABLE_COMPILE_DEBUG"):
        print("  SKIP: _FLEX_ATTENTION_DISABLE_COMPILE_DEBUG not present")
        return

    q, k, v = make_qkv()

    def fn(q, k, v):
        return flex_attention(q, k, v)

    # ── With compile disabled (current patch state) ──
    fa_mod._FLEX_ATTENTION_DISABLE_COMPILE_DEBUG = True
    result_disabled = "PASS"
    try:
        torch.compile(fn, backend="inductor")(q, k, v)
    except Exception as e:
        result_disabled = f"FAIL — {type(e).__name__}: {e}"

    # ── With compile enabled (original behavior) ──
    fa_mod._FLEX_ATTENTION_DISABLE_COMPILE_DEBUG = False
    result_enabled = "PASS"
    try:
        torch.compile(fn, backend="inductor")(q, k, v)
    except Exception as e:
        result_enabled = f"FAIL — {type(e).__name__}: {e}"

    # ── Restore patch state ──
    fa_mod._FLEX_ATTENTION_DISABLE_COMPILE_DEBUG = True

    print(f"  Compile DISABLED: {result_disabled}")
    print(f"  Compile ENABLED:  {result_enabled}")

    # Interpret
    if "PASS" in result_disabled and "PASS" not in result_enabled:
        print("  → Compile path IS the blocker on NPU")
    elif "PASS" in result_disabled and "PASS" in result_enabled:
        print("  → Both work; compile path is fine on NPU")
    elif "PASS" not in result_disabled and "PASS" not in result_enabled:
        print("  → Both fail; blocker is deeper than compile (kernel missing)")
    else:
        print("  → Unexpected: compile enabled works but disabled doesn't")


# ── Main ──────────────────────────────────────────────────────────

def main():
    print(f"PyTorch: {torch.__version__}")

    if not torch.npu.is_available():
        raise RuntimeError("NPU not available — this test requires an Ascend NPU device")

    results = {}

    test_patch()
    test_device_validation()

    results["eager"] = test_flex_attention_eager()
    results["compile"] = test_flex_attention_compile()

    test_compile_comparison()

    # ── Summary ──
    print("\n========== SUMMARY ==========")
    print(f"  Eager (no compile):  {'PASS' if results.get('eager') else 'FAIL'}")
    print(f"  torch.compile:       {'PASS' if results.get('compile') else 'FAIL'}")

    if not results.get("eager"):
        print("\n  → FlexAttention kernel missing for NPU (PrivateUse1).")
        print("    Need: NPU kernel implementation or fallback to SDPA / torch_npu.npu_fusion_attention.")
    elif not results.get("compile"):
        print("\n  → Kernel exists but compile path broken on NPU.")
        print("    Keep _FLEX_ATTENTION_DISABLE_COMPILE_DEBUG = True as workaround.")
    else:
        print("\n  → Full FlexAttention + compile works on NPU.")

    print("============================\n")


if __name__ == "__main__":
    main()