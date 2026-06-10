"""Fresh binders for F-A v2 architectural-transfer holdout.

Families:
    A — U-Net direct: ResNet-18 encoder (no ImageNet pretrain) + U-Net decoder.
    B — ConvNeXt+FPN: ConvNeXt-Tiny (ImageNet pretrain) + FPN decoder.
    C — ResNet+pyramid: ResNet-50 (ImageNet pretrain) + dilated-pyramid decoder.
    D — phase-aware (HELD OUT): DoG-augmented input + small CNN, no pretrain.

All take (B, 4, capture_h, capture_w) packed CFA in [0, 1], output
(B, 3, 1080, 1920) RGB in [0, 1] via sigmoid.
"""
from .family_a_unet_direct import FreshBinderA  # noqa: F401
from .family_b_convnext_fpn import FreshBinderB  # noqa: F401
from .family_c_resnet_pyramid import FreshBinderC  # noqa: F401
from .family_d_phase_aware import FreshBinderD  # noqa: F401
from .family_d_large import FreshBinderDLarge  # noqa: F401

# Public registry for the unified launcher
FRESH_BINDER_REGISTRY = {
    "A": FreshBinderA,
    "B": FreshBinderB,
    "C": FreshBinderC,
    "D": FreshBinderD,
    "D_large": FreshBinderDLarge,
}
