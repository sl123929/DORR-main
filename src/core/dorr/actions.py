"""Action-shape checks for the DORR integration.

ROLeR's recommender environment declares a one-element Box action space, while
the actor emits one catalog item ID per environment.  This adapter keeps that
existing scalar-ID contract explicit at the vector-environment boundary.
"""

from __future__ import annotations

from typing import Any

import numpy as np


def normalize_discrete_actions(action: Any, env_ids: Any, catalog_size: int) -> np.ndarray:
    """Return one legal integer item ID for every environment in ``env_ids``.

    A one-element action dimension is flattened because the environment worker
    receives one scalar item ID per environment.  Arrays containing logits,
    embeddings, or another number of actions are rejected explicitly.
    """
    env_count = len(env_ids)
    array = np.asarray(action)
    original_shape = array.shape
    if array.ndim == 0:
        array = array.reshape(1)
    if array.size != env_count:
        raise ValueError(
            "DORR actor must emit exactly one item ID per environment: "
            f"action_shape={original_shape}, action_size={array.size}, "
            f"env_count={env_count}, env_ids_shape={np.asarray(env_ids).shape}"
        )
    array = array.reshape(-1)
    if not np.issubdtype(array.dtype, np.integer):
        if not np.isfinite(array).all() or not np.equal(array, np.round(array)).all():
            raise ValueError(
                "DORR actor action must contain finite integer item IDs; "
                f"got dtype={array.dtype}, shape={original_shape}"
            )
    result = array.astype(np.int64, copy=False)
    if (result < 0).any() or (result >= int(catalog_size)).any():
        raise ValueError(
            "DORR actor emitted an item ID outside the legal catalog: "
            f"min={int(result.min())}, max={int(result.max())}, "
            f"catalog_size={int(catalog_size)}"
        )
    return result
