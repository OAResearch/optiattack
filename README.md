[![Run Tests](https://github.com/OAResearch/optiattack/actions/workflows/ci.yml/badge.svg)](https://github.com/OAResearch/optiattack/actions/workflows/ci.yml)

# OptiAttack

OptiAttack (OA) is an optimization-based adversarial example generation framework designed for network test generation. It provides a modular, extensible platform for generating adversarial examples using evolutionary and search-based algorithms, with a focus on image-based attacks. The project includes both a core engine and a client-server architecture for remote testing, as well as a modern web UI for interactive use.

---

## Table of Contents

- [Features](#features)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Web UI](#web-ui)
- [Client Usage](#client-usage)
- [Configuration](#configuration)
- [Project Structure](#project-structure)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgements](#acknowledgements)

---

## Features

- **Optimization-based adversarial example generation** for network testing.
- **Modular architecture**: Easily extend algorithms, fitness functions, and mutators.
- **Remote controller**: Test against remote models or systems under test (NUT).
- **Web UI**: User-friendly Gradio interface for interactive attack generation and visualization.
- **Comprehensive logging and reporting**: Output images, statistics, and confidence charts.
- **Configurable parameters**: Fine-tune every aspect of the search and attack process.

---

## Installation

### Prerequisites

- Python 3.9 or higher
- pip

### Install Core and UI

```bash
pip install -r requirements.txt
```

### Install Client (for remote NUT integration)

```bash
cd client
pip install -e .
```

---

## Quick Start

### Command-Line Usage

To run OptiAttack from the command line:

```bash
python main.py --input_image path/to/image.jpg --nut_host localhost --nut_port 38000 --max_evaluations 1000
```

**Common arguments:**
- `--input_image`: Path to the input image.
- `--nut_host`: Host address of the network under test (NUT).
- `--nut_port`: Port number of the NUT.
- `--max_evaluations`: Maximum number of search evaluations.

For a full list of parameters, see [Configuration](#configuration).

---

## Web UI

OptiAttack includes a Gradio-based web interface for interactive use.

### Launch the Web UI

```bash
python main.py --enable_ui True
```

- Upload an input image.
- Set parameters (host, port, image size, max evaluations, etc.).
- Click "Run OptiAttack" to start the attack.
- View results, confidence charts, and reports directly in the browser.

---

## Client Usage

The `optiattack_client` package provides a FastAPI-based server for integrating with remote models or systems under test (NUT).

### Example

```python
from optiattack_client import collect_info

@collect_info(host="localhost", port=38000)
def predict(image_array):
    # Your prediction logic here
    return {"predictions": ...}
```

- The client exposes endpoints for running attacks, getting info, and managing state.
- See `client/optiattack_client.py` for details.

---

## Configuration

OptiAttack is highly configurable. You can set parameters via command-line arguments or configuration files.

### Key Parameters

| Parameter                | Default         | Description                                              |
|--------------------------|----------------|----------------------------------------------------------|
| `algorithm`              | mio            | Search algorithm for optimization                        |
| `input_image`            | ./tests/test_img.jpeg | Path to the input image                           |
| `image_width`            | 224            | Image width (pixels)                                     |
| `image_height`           | 224            | Image height (pixels)                                    |
| `max_evaluations`        | 1000           | Maximum number of search evaluations                     |
| `nut_host`               | localhost      | Host address for the NUT                                 |
| `nut_port`               | 38000          | Port number for the NUT                                  |
| `enable_ui`              | False          | Enable the web UI                                        |
| `enable_pruning`         | False          | Enable pruning of final results                          |
| ...                      | ...            | See [docs/parameters.md](docs/parameters.md) for all parameters |

---

## Project Structure

```
.
├── main.py                # Entry point for the core application
├── gradio_ui.py           # Web UI implementation
├── core/                  # Core logic: algorithms, problem definitions, services
├── client/                # Client package for remote NUT integration
├── docs/                  # Documentation (parameters, usage, etc.)
├── requirements.txt       # Python dependencies
└── tests/                 # Test cases and test images
```

---

## Contributing

Contributions are welcome! Please open issues or pull requests for bug fixes, new features, or documentation improvements.

1. Fork the repository.
2. Create a new branch for your feature or fix.
3. Ensure code passes linting and tests.
4. Submit a pull request with a clear description.

---

## License

This project is licensed under the GNU Lesser General Public License v3 (LGPLv3). See the [LICENSE](LICENSE) file for details.

---

## Acknowledgements

- Developed by OptiAttack Team.
- Built with FastAPI, Gradio, and other open-source libraries.

---

**For more details, see the [docs/parameters.md](docs/parameters.md) file and in-code documentation.**
