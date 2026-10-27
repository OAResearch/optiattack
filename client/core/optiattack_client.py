import base64
from typing import Callable, List, Optional

import numpy as np
import uvicorn

from fastapi import FastAPI, HTTPException
from functools import wraps
import threading

from pydantic import BaseModel

try:
    from client.core import constants
except ImportError:
    import constants


class MatrixModel(BaseModel):
    image: str
    target: Optional[str] = None


class GradientModel(BaseModel):
    image: str
    shape: List[int]
    label: str
    kind: str


GRADIENT_KINDS = ("loss", "class")


def _decode_gradient_image(data: GradientModel) -> np.ndarray:
    """Decode and validate the float32 image of a gradient request (HTTP 400 on failure)."""
    if len(data.shape) != 3 or any(dim <= 0 for dim in data.shape):
        raise HTTPException(status_code=400, detail=f"shape must be [H, W, C] with positive sizes, got {data.shape}")
    try:
        raw = base64.b64decode(data.image, validate=True)
    except Exception:
        raise HTTPException(status_code=400, detail="image is not valid base64")
    if len(raw) % 4 != 0:
        raise HTTPException(status_code=400, detail="image must be float32 bytes (length not a multiple of 4)")
    expected = int(np.prod(data.shape))
    if expected != len(raw) // 4:
        raise HTTPException(
            status_code=400,
            detail=f"shape {data.shape} has {expected} values but image holds {len(raw) // 4} float32 values")
    return np.frombuffer(raw, dtype=np.float32).reshape(data.shape).copy()


def collect_info(host: str = constants.DEFAULT_CONTROLLER_HOST,
                 port: int = constants.DEFAULT_CONTROLLER_PORT,
                 labels: Optional[List[str]] = None,
                 gradient_fn: Optional[Callable[[np.ndarray, str, str], np.ndarray]] = None):
    """
    Decorator generator that:
    - Tracks method calls.
    - Automatically starts a FastAPI application.

    Args:
        host (str): Host where the FastAPI app will run.
        port (int): Port where the FastAPI app will run.
        labels (Optional[List[str]]): Class names in model output order (reported to OptiAttack).
        gradient_fn (Optional[Callable]): White-box hook ``gradient_fn(image, label, kind) -> gradient``.
            ``image`` is float32 HxWxC on the 0-255 scale, ``kind`` is "loss" (cross-entropy w.r.t. ``label``)
            or "class" (logit of ``label``); the returned gradient must be float32 with the image's shape.
            When set, "gradients" is added to the reported capabilities and /api/v1/gradientNut is served.
            The decorated function may also return "logits" (one value per label, in ``labels`` order).

    Raises:
        ValueError: if ``gradient_fn`` is given without ``labels`` (white-box use needs the class list).
    """
    if gradient_fn is not None and not labels:
        raise ValueError("gradient_fn needs labels: pass labels=[...] (class names in model output order) "
                         "to collect_info for white-box use")

    def decorator(func):
        # State dictionary to track method call counts
        state = {
            "is_running": False,
            "target": None,
            "controller_host": "",
            "controller_port": None,
            "run_nut": constants.RUN_NUT_PATH,
            "stop_nut": constants.STOP_NUT_PATH,
            "info_nut": constants.INFO_NUT_PATH,
            "new_action": constants.NEW_ACTION,
            "labels": labels,
            "capabilities": ["scores", "gradients"] if gradient_fn is not None else ["scores"],
        }
        app = FastAPI()

        @wraps(func)
        async def wrapper(*args, **kwargs):
            return await func(*args, **kwargs)

        @app.get("/")
        async def test():
            return {"message": "Hello, FastAPI is running!"}

        @app.get(f"{constants.INFO_NUT_PATH}")
        async def get_nut_state():
            return state

        @app.post(f"{constants.RUN_NUT_PATH}")
        async def run_nut(data: MatrixModel):
            state["is_running"] = True
            state["controller_host"] = host
            state["controller_port"] = port
            state["target"] = data.target
            image_base64 = base64.b64decode(data.image)
            array_data = np.frombuffer(image_base64, np.uint8)
            additional_data = {
                "target": data.target,
            }
            result = func(array_data, additional_data)
            state["predictions"] = result["predictions"]
            if "logits" in result:
                state["logits"] = result["logits"]
            else:
                state.pop("logits", None)

            return state

        @app.post(f"{constants.STOP_NUT_PATH}")
        async def stop_nut():
            state["is_running"] = False
            state["controller_host"] = ""
            state["controller_port"] = None
            return state

        @app.post(f"{constants.NEW_ACTION}")
        async def new_action(data: MatrixModel):
            image_base64 = base64.b64decode(data.image)
            array_data = np.frombuffer(image_base64, np.uint8)
            additional_data = {
                "target": state["target"]
            }
            return func(array_data, additional_data)

        @app.post(f"{constants.GRADIENT_NUT}")
        async def gradient_nut(data: GradientModel):
            if gradient_fn is None:
                raise HTTPException(status_code=400, detail="this NUT does not provide gradient_fn (no gradients)")
            if data.kind not in GRADIENT_KINDS:
                raise HTTPException(status_code=400, detail=f"unknown kind '{data.kind}', expected one of "
                                                            f"{list(GRADIENT_KINDS)}")
            if not labels or data.label not in labels:
                raise HTTPException(status_code=400, detail=f"unknown label '{data.label}', known labels: {labels}")
            image = _decode_gradient_image(data)
            gradient = np.asarray(gradient_fn(image, data.label, data.kind), dtype=np.float32)
            if gradient.shape != image.shape:
                raise HTTPException(
                    status_code=400,
                    detail=f"gradient_fn returned shape {list(gradient.shape)}, expected {list(image.shape)}")
            gradient = np.ascontiguousarray(gradient)
            return {"gradient": base64.b64encode(gradient.tobytes()).decode(), "shape": list(gradient.shape)}

        def start_server():
            uvicorn.run(app, host=host, port=port)

        wrapper.app = app
        threading.Thread(target=start_server, daemon=True).start()

        return wrapper

    return decorator
