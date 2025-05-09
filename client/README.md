# OptiAttack Client

The OptiAttack Client (`optiattack_client`) is a FastAPI-based server package designed to facilitate remote integration between the OptiAttack core engine and external models or systems under test (NUT). It provides a simple decorator and API endpoints for seamless communication and adversarial test execution.

---

## Table of Contents

- [Features](#features)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [API Endpoints](#api-endpoints)
- [Example Usage](#example-usage)
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

2. **Start your script**. The FastAPI server will automatically run and expose endpoints for OptiAttack to communicate with.

---

## API Endpoints

The client exposes the following endpoints (default base path: `/api/v1`):

- `GET /api/v1/infoNUT` — Get the current state of the NUT
- `POST /api/v1/runNUT` — Run a prediction on the NUT (expects base64-encoded image)
- `POST /api/v1/stopNUT` — Stop the NUT
- `POST /api/v1/newAction` — Apply a new action (mutation) and get prediction

See `client/constants.py` for all endpoint paths and defaults.

---

## Example Usage

```python
from optiattack_client import collect_info

@collect_info(host="localhost", port=38000)
def predict(image_array):
    # Example: Use your ML model to predict
    # result = model.predict(image_array)
    return {"predictions": [0.1, 0.9]}  # Example output

# The FastAPI server is now running at http://localhost:38000
```

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
