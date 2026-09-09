"""Shared scene-slot validation and explicit checkpoint initialization policy."""
import logging

import torch
from torch import nn

logger = logging.getLogger(__name__)


def scene_count(enabled, count):
    if not isinstance(enabled, bool):
        raise ValueError("scene_tokens_enabled must be a bool")
    if isinstance(count, bool) or not isinstance(count, int) or count <= 0:
        raise ValueError("num_scene_tokens must be a positive integer")
    return count if enabled else 0


def make_scene_tokens(count, dim):
    if isinstance(count, bool) or not isinstance(count, int) or count < 0:
        raise ValueError("num_scene_tokens must be a non-negative integer in the backbone")
    if count == 0:
        return None
    return nn.Parameter(torch.empty(count, dim).normal_(std=0.02))


def load_scene_state(model, state_dict, allow_base_init=False):
    """Only explicitly allow the new slots/projection to be absent in a base checkpoint.

    Validate before loading so malformed checkpoints cannot partly overwrite a model.
    Ordinary scene checkpoints must have an exact key/shape match.
    """
    state = {k.removeprefix("module.").replace("_orig_mod.", ""): v for k, v in state_dict.items()}
    if len(state) != len(state_dict):
        raise RuntimeError("Duplicate checkpoint keys after DDP/compile prefix normalization")
    expected = model.state_dict()
    missing = set(expected) - set(state)
    unexpected = set(state) - set(expected)
    additions = {
        k for k in expected if k.endswith(".scene_tokens") or k.startswith("scene_color_query.")
    }
    allowed = additions if allow_base_init else set()
    # A partial scene checkpoint is never treated as a base initialization.
    valid_base = bool(additions) and missing == additions and not (set(state) & additions)
    mismatched = [k for k in set(expected) & set(state) if expected[k].shape != state[k].shape]
    if unexpected or mismatched or (missing and not (valid_base and missing <= allowed)):
        raise RuntimeError(
            f"Incompatible scene checkpoint: missing={sorted(missing)}, "
            f"unexpected={sorted(unexpected)}, shape_mismatch={sorted(mismatched)}. "
            "Use explicit scene_token_init_from_base=True only for an intact base checkpoint."
        )
    result = model.load_state_dict(state, strict=not missing)
    if missing:
        logger.warning("Base initialization; new untrained parameters: %s", sorted(missing))
    return result
