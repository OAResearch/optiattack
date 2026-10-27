"""Optional PyTorch helpers that build NUT functions for the white-box OptiAttack client API."""
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

INSTALL_HINT = "pip install optiattack_client[torch]"
_KINDS = ("loss", "class")


def _import_torch():
    """Import torch lazily; raise ImportError with an install hint when it is missing."""
    try:
        import torch
    except ImportError as exc:
        raise ImportError(INSTALL_HINT) from exc
    return torch


def _infer_shape(size: int) -> Tuple[int, int, int]:
    """Guess HxWxC from a flat buffer size (square image, 3 channels first, then 1)."""
    for channels in (3, 1):
        if size % channels == 0:
            side = int(round((size // channels) ** 0.5))
            if side * side * channels == size:
                return side, side, channels
    raise ValueError(f"cannot infer the image shape from a buffer of {size} values; pass input_shape=(H, W, C)")


def _check_logits(logits, labels: Sequence[str]):
    """Return the 1-D logit vector and verify it has one entry per label."""
    logits = logits.reshape(-1)
    if logits.shape[0] != len(labels):
        raise ValueError(f"model returned {logits.shape[0]} logits but {len(labels)} labels were given")
    return logits


def _top_k_predictions(probs: np.ndarray, labels: Sequence[str], top_k: int,
                       target: Optional[str] = None) -> List[Dict[str, Any]]:
    """Top-k ``[{"label", "score"}]`` in descending score order.

    A ``target`` label outside the top-k replaces the lowest top-k entry (as the benchmark services do), so a
    targeted attack always sees the target's score.
    """
    order = [int(i) for i in np.argsort(-probs, kind="stable")[:top_k]]
    if target is not None:
        if target not in labels:
            raise ValueError(f"Target class '{target}' is not in the class list.")
        target_index = labels.index(target)
        if target_index not in order:
            order[-1] = target_index
    return [{"label": labels[i], "score": float(probs[i])} for i in order]


def torch_predict_fn(model: Any, preprocess: Callable, labels: Sequence[str], top_k: int = 5,
                     input_shape: Optional[Tuple[int, int, int]] = None) -> Callable:
    """Build a NUT function returning the top-k softmax ``predictions`` and the full ``logits`` vector.

    ``predictions`` is ``[{"label": ..., "score": ...}]`` in descending score order; a ``target`` in
    ``additional_data`` that is not in the top-k replaces the lowest entry. ``preprocess`` maps a float32
    HxWxC (0-255) torch tensor to the batched model input (e.g. 1xCxHxW) and must be the same function given
    to ``torch_gradient_fn``. ``input_shape`` is the (H, W, C) of the flat uint8 buffer; if omitted it is
    inferred for square images with 3 or 1 channels. The model should already be in eval mode.
    """
    torch = _import_torch()
    labels = list(labels)

    def fn(data, additional_data=None) -> Dict[str, Any]:
        shape = tuple(input_shape) if input_shape is not None else _infer_shape(int(np.size(data)))
        image = torch.from_numpy(np.asarray(data, dtype=np.float32).reshape(shape).copy())
        with torch.no_grad():
            logits = _check_logits(model(preprocess(image)), labels).double()
        probs = torch.softmax(logits, dim=0).cpu().numpy()
        target = additional_data.get("target") if additional_data else None
        return {"predictions": _top_k_predictions(probs, labels, top_k, target),
                "logits": logits.cpu().tolist()}

    return fn


def torch_gradient_fn(model: Any, preprocess: Callable, labels: Sequence[str]) -> Callable:
    """Build a ``gradient_fn(image, label, kind)`` for ``collect_info`` using torch autograd.

    ``preprocess`` must be differentiable: float32 HxWxC (0-255) tensor -> batched model input.
    ``kind`` "loss" is the cross-entropy w.r.t. ``label``; "class" is the logit of ``label``.
    The model should already be in eval mode.
    """
    torch = _import_torch()
    labels = list(labels)

    def fn(image: np.ndarray, label: str, kind: str) -> np.ndarray:
        if kind not in _KINDS:
            raise ValueError(f"unknown kind '{kind}', expected one of {list(_KINDS)}")
        if label not in labels:
            raise ValueError(f"unknown label '{label}'")
        index = labels.index(label)
        with torch.enable_grad():
            x = torch.from_numpy(np.asarray(image, dtype=np.float32).copy()).requires_grad_(True)
            logits = _check_logits(model(preprocess(x)), labels)
            if kind == "class":
                objective = logits[index]
            else:
                objective = torch.nn.functional.cross_entropy(logits.unsqueeze(0), torch.tensor([index]))
            (grad,) = torch.autograd.grad(objective, x)
        return grad.detach().cpu().numpy().astype(np.float32)

    return fn
