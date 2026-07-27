import functools


def apply_patch():

    from torch.nn.attention import flex_attention as fa_mod


    if getattr(fa_mod, "_npu_patch_applied", False):
        return


    # ---------------------------------------
    # Disable FlexAttention compile path
    # ---------------------------------------
    if hasattr(
        fa_mod,
        "_FLEX_ATTENTION_DISABLE_COMPILE_DEBUG"
    ):
        fa_mod._FLEX_ATTENTION_DISABLE_COMPILE_DEBUG = True

        print(
            "[Patch] "
            "Disabled FlexAttention compile debug path"
        )


    # ---------------------------------------
    # Patch device validation
    # ---------------------------------------
    original_validate = fa_mod._validate_device


    @functools.wraps(original_validate)
    def npu_validate(query, key, value):

        if query.device.type == "npu":
            print(
                "[Patch] "
                "Skipping device validation for NPU"
            )
            return

        return original_validate(
            query,
            key,
            value
        )


    fa_mod._validate_device = npu_validate


    fa_mod._npu_patch_applied = True


    print(
        "[Patch] FlexAttention NPU patch installed"
    )


apply_patch()
