# OptiAttack Client

The OptiAttack Client (`optiattack_client`) is a FastAPI-based server package designed to facilitate remote integration between the OptiAttack core engine and external models or systems under test (NUT). It provides a simple decorator and API endpoints for seamless communication and adversarial test execution.

---

## Table of Contents

- [Features](#features)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [API Endpoints](#api-endpoints)
- [Example Usage](#example-usage)
- [White-box (gradients)](#white-box-gradients)
- [Configuration](#configuration)
- [Contributing](#contributing)
- [License](#license)

---

## Features

- **FastAPI server** for remote model or NUT integration
- **Decorator-based interface** for easy wrapping of prediction functions
- **Automatic endpoint creation** for running, stopping, and querying the NUT
- **Supports image-based adversarial attacks**
- **Easy integration with OptiAttack core**

---

## Installation

You can install the OptiAttack Client package using pip:

```bash
pip install optiattack-client
```

Alternatively, if you want to install from the local directory (for development or custom modifications), run:

```bash
cd client
pip install -e .
```

---

## Quick Start

1. **Wrap your prediction function** with the provided decorator:

```python
from optiattack_client import collect_info

@collect_info(host="localhost", port=38000)
def predict(image_array):
    # Your prediction logic here
    return {"predictions": ...}
```

2. **(Optional) Report the class list** with `labels=`, in the same order as the model's output scores:

```python
@collect_info(host="localhost", port=38000, labels=["zebra", "horse"])
def predict(image_array):
    ...
```

   `labels` is required for OptiAttack baseline attacks (`--algorithm baseline`), which need the full class list to build probability vectors. Without it the client behaves as before and reports `labels: null`. `infoNUT` also reports `capabilities` (currently `["scores"]`).

3. **Start your script**. The FastAPI server will automatically run and expose endpoints for OptiAttack to communicate with.

---

## API Endpoints

The client exposes the following endpoints (default base path: `/api/v1`):

- `GET /api/v1/infoNUT` — Get the current state of the NUT
- `POST /api/v1/runNUT` — Run a prediction on the NUT (expects base64-encoded image)
- `POST /api/v1/stopNUT` — Stop the NUT
- `POST /api/v1/newAction` — Apply a new action (mutation) and get prediction
- `POST /api/v1/gradientNut` — Image gradient (white-box; only when `gradient_fn` is given)

See `client/constants.py` for all endpoint paths and defaults.

---

## Example Usage

```python
from optiattack_client import collect_info

@collect_info(host="localhost", port=38000, labels=["cat", "dog"])
def predict(image_array, additional_data=None):
    # Example: Use your ML model to predict
    # result = model.predict(image_array)
    # One {"label", "score"} entry per reported class, highest score first
    return {"predictions": [{"label": "dog", "score": 0.9}, {"label": "cat", "score": 0.1}]}

# The FastAPI server is now running at http://localhost:38000
```

---

## White-box (gradients)

Pass an optional `gradient_fn` to expose image gradients to OptiAttack's white-box attacks
(`collect_info(host, port, labels=None, gradient_fn=None)`). Without it nothing changes.
With it, the NUT reports `capabilities == ["scores", "gradients"]` and serves `POST /api/v1/gradientNut`.

`gradient_fn(image, label, kind) -> gradient`:

- `image`: `np.ndarray`, float32, HxWxC, raw pixel scale 0-255 (it is **not** rounded to uint8; keep any
  normalisation inside your function).
- `label`: a class name from `labels` (`labels` is required for white-box use; `gradient_fn` without `labels`
  raises `ValueError` when the function is decorated).
- `kind`: `"loss"` is the gradient of the cross-entropy w.r.t. `label`; `"class"` is the gradient of the
  **logit** of `label`.
- returns a float32 array of the same shape as `image` (gradient in pixel space).

Endpoint contract: request `{"image": base64(float32 bytes), "shape": [H, W, C], "label": str, "kind": "loss"|"class"}`,
response `{"gradient": base64(float32 bytes), "shape": [H, W, C]}`. The server answers HTTP 400 with a `detail`
message when there is no `gradient_fn`, the image is not valid base64 float32 bytes, `shape` does not match the
payload, `kind` or `label` is unknown, or the returned gradient has a different shape.

**Logits are required for white-box use.** A NUT that declares `"gradients"` must also return, next to
`predictions`, a `logits` list with the scores of **all** classes in `labels` order. The client passes it
through unchanged (`newAction` response and `runNUT` state). Black-box runs ignore `logits`.

PyTorch helpers (`pip install optiattack_client[torch]`; torch is only imported when they are called):

```python
import torch
from optiattack_client import collect_info
from torch_support import torch_predict_fn, torch_gradient_fn

labels = ["cat", "dog"]
model = torch.nn.Sequential(...).eval()

def preprocess(x):  # float32 HxWxC (0-255), differentiable -> batched model input, e.g. 1xCxHxW
    return ((x.permute(2, 0, 1).unsqueeze(0) / 255.0) - 0.5) / 0.5

predict = torch_predict_fn(model, preprocess, labels, top_k=5)
# -> {"predictions": [{"label": "dog", "score": 0.93}, ...] (top-k softmax, highest first), "logits": [all classes]}
gradient = torch_gradient_fn(model, preprocess, labels)

@collect_info(host="localhost", port=38000, labels=labels, gradient_fn=gradient)
def nut(image_array, additional_data=None):
    return predict(image_array, additional_data)
```

`torch_predict_fn` reshapes the flat uint8 buffer to HxWxC; for square images with 3 (or 1) channels this is
inferred, otherwise pass `input_shape=(H, W, C)`. Put the model in `eval()` mode yourself.
In targeted runs a `target` (from `additional_data`) outside the top-k replaces the lowest top-k entry, so its
score is always reported; an unknown `target` raises `ValueError`.

The helpers create CPU tensors: for a CUDA model, move the input inside `preprocess` (e.g. `return x.to(device)`).

---

## Configuration

- **Host and Port**: Set via the `collect_info` decorator arguments.
- **Endpoints**: Configurable in `client/constants.py`.
- **Python Version**: Requires Python 3.9 or higher.

---

## Contributing

Contributions are welcome! Please open issues or pull requests for bug fixes, new features, or documentation improvements.

---

## License

This package is part of the OptiAttack project and is licensed under the GNU Lesser General Public License v3 (LGPLv3).
