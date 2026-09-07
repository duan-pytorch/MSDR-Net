# -*- coding: utf-8 -*-
"""
MSDR-Net private (redacted) key parameters.

--------------------------------------------------------------------------
NOTE FOR USERS OF THIS PUBLIC RELEASE
--------------------------------------------------------------------------
Several key hyperparameters of MSDR-Net are INTENTIONALLY REDACTED in this
public repository (they are set to ``None`` below). The released code runs
out of the box, because every redacted value automatically falls back to a
reasonable placeholder value (shown in ``PLACEHOLDER_FALLBACKS``).

If you need the exact key parameters used in the paper experiments
(e.g., the adaptive kernel mapping constants of the ECA module, the MLP
head configuration, class-reweighting factors, and the learning-rate
schedule details), please contact the author:

    Email: a13995115025@163.com

Please do NOT treat the placeholder fallback values as the exact values
reported in the manuscript; they are sensible defaults only.
--------------------------------------------------------------------------
"""

AUTHOR_CONTACT = "a13995115025@163.com"

# =========================================================================
# REDACTED KEY PARAMETERS (set to None on purpose -- see note above)
# =========================================================================

# --- ECA (Efficient Channel Attention) adaptive kernel mapping ----------
# Kernel size k is adaptively mapped from the channel number C via
#     k = psi(C) = | log2(C) / GAMMA + BETA / GAMMA |_{odd}
# The exact mapping constants are redacted in this public release.
ECA_GAMMA = None          # [REDACTED] contact the author for the exact value
ECA_BETA = None           # [REDACTED] contact the author for the exact value

# --- Classification head (MLP) ------------------------------------------
# Hidden dimension and dropout of the MLP head (GAP -> MLP -> Softmax).
MLP_HIDDEN_DIM = None     # [REDACTED] contact the author for the exact value
MLP_DROPOUT = None        # [REDACTED] contact the author for the exact value

# --- Class re-weighting for the weighted cross-entropy loss -------------
# Weights are in principle computed from the inverse class frequency of the
# training split; the exact constants used in the paper are redacted.
CLASS_WEIGHT_NEG = None   # [REDACTED] benign-class weight in the paper
CLASS_WEIGHT_POS = None   # [REDACTED] malignant-class weight in the paper

# --- Optimization schedule details --------------------------------------
# Exact initial learning rate / weight decay / LR-decay milestones.
INIT_LR = None            # [REDACTED] contact the author for the exact value
WEIGHT_DECAY = None       # [REDACTED] contact the author for the exact value
LR_MILESTONES = None      # [REDACTED] e.g. list of epochs at which LR decays
LR_GAMMA = None           # [REDACTED] multiplicative LR decay factor

# =========================================================================
# PLACEHOLDER FALLBACKS (reasonable mock values; NOT the paper values)
# =========================================================================
# These fallbacks keep the released code fully runnable. They are chosen to
# be standard, sensible choices for this model family.
PLACEHOLDER_FALLBACKS = {
    "ECA_GAMMA": 2.0,          # standard ECA-Net style mapping constant
    "ECA_BETA": 1.0,           # standard ECA-Net style mapping constant
    "MLP_HIDDEN_DIM": 256,     # two-layer MLP head, e.g. 1024 -> 256 -> 2
    "MLP_DROPOUT": 0.5,        # dropout inside the MLP head
    "CLASS_WEIGHT_NEG": 0.64,  # approx. inverse-frequency weight (benign)
    "CLASS_WEIGHT_POS": 2.27,  # approx. inverse-frequency weight (malignant)
    "INIT_LR": 1e-4,           # AdamW initial learning rate
    "WEIGHT_DECAY": 1e-2,      # AdamW weight decay
    "LR_MILESTONES": [80, 140],  # MultiStepLR milestones (epochs)
    "LR_GAMMA": 0.1,           # MultiStepLR decay factor
}

_REDACTED_NOTICE_EMITTED = set()


def resolve(name):
    """Return the effective value of a key parameter.

    If the parameter has been filled in (i.e. is not ``None``), the real
    value is used. Otherwise a reasonable placeholder fallback is returned
    and a one-time notice is printed.
    """
    if name not in PLACEHOLDER_FALLBACKS:
        raise KeyError(f"Unknown private parameter: {name!r}")

    value = globals().get(name, None)
    if value is not None:
        return value

    if name not in _REDACTED_NOTICE_EMITTED:
        _REDACTED_NOTICE_EMITTED.add(name)
        print(
            f"[MSDR-Net] Key parameter '{name}' is redacted in this public "
            f"release; using a placeholder value "
            f"({PLACEHOLDER_FALLBACKS[name]!r}). For the exact value used in "
            f"the paper, please contact: {AUTHOR_CONTACT}"
        )
    return PLACEHOLDER_FALLBACKS[name]
