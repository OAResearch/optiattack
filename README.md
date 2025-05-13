![GitHub License](https://img.shields.io/github/license/oaresearch/optiattack)
[![Run Tests](https://github.com/OAResearch/optiattack/actions/workflows/ci.yml/badge.svg)](https://github.com/OAResearch/optiattack/actions/workflows/ci.yml)
[![PyPI version](https://badge.fury.io/py/optiattack-client.svg)](https://badge.fury.io/py/optiattack-client)
[![Docker Pulls](https://img.shields.io/docker/pulls/oaresearch/optiattack.svg)](https://hub.docker.com/r/oaresearch/optiattack)
![Docker Image Version](https://img.shields.io/docker/v/oaresearch/optiattack)
![Docker Image Size](https://img.shields.io/docker/image-size/oaresearch/optiattack)
[![PyPI Downloads](https://static.pepy.tech/badge/optiattack-client)](https://pepy.tech/projects/optiattack-client)
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
## Docker Usage

OptiAttack can be run using Docker, which provides an isolated environment for running the application. You can either pull the pre-built image from Docker Hub or build it locally.

### Pulling from Docker Hub

The easiest way to get started is to pull the pre-built image from Docker Hub:

```bash
docker pull oaresearch/optiattack
```

### Running the Container

After pulling the image, you can run it with:

```bash
docker run -v /path/to/images:/app/images -v /path/to/output:/app/output oaresearch/optiattack [options]
```

### Volume Mappings

The Docker container requires two volume mappings:
- `/app/images`: Directory containing input images
- `/app/output`: Directory for storing output results

### Example Command

```bash
docker run --rm \
    -v $(pwd)/images/:/app/images \
    -v $(pwd)/output:/app/output \
    oaresearch/optiattack \
    --input_image ./images/test_img.jpeg \
    --nut_host host.docker.internal \
    --seed 8
```
To run the container with the web UI:

```bash
docker run --rm \
    -v $(pwd)/images/:/app/images \
    -v $(pwd)/output:/app/output \
    oaresearch/optiattack \
    --enable_ui True
```

### Important Notes

1. **Volume Paths**:
   - Use absolute paths for volume mappings
   - Windows paths should use forward slashes (/) or escaped backslashes (\\)
   - The paths should exist on your host machine

2. **Network Access**:
   - Use `host.docker.internal` to access services running on your host machine
   - This is particularly important for the NUT (Network Under Test) connection

3. **Common Parameters**:
   - `--input_image`: Path to the input image (relative to the mounted images directory)
   - `--nut_host`: Host address for the NUT (use `host.docker.internal` for local services)
   - `--seed`: Random seed for reproducibility
   - Other parameters can be added as needed

### Building Locally

If you need to build the Docker image locally instead of pulling from Docker Hub:

```bash
docker-compose build
```

### Running with Docker Compose

For development or testing, you can use Docker Compose:

```bash
docker-compose up
```

This will use the configuration from `docker-compose.yml` and automatically set up the required volumes and environment variables.
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
## Funding

This work was supported by the Erciyes University Scientific Research Fund (ERU-BAP, Project No: FBA-2024-13536).

---

## Acknowledgements

- Developed by OptiAttack Team.
- Built with FastAPI, Gradio, and other open-source libraries.

---

**For more details, see the [docs/parameters.md](docs/parameters.md) file and in-code documentation.**
