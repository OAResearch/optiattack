![GitHub License](https://img.shields.io/github/license/oaresearch/optiattack)
[![Run Tests](https://github.com/OAResearch/optiattack/actions/workflows/ci.yml/badge.svg)](https://github.com/OAResearch/optiattack/actions/workflows/ci.yml)
[![PyPI version](https://badge.fury.io/py/optiattack-client.svg)](https://badge.fury.io/py/optiattack-client)
[![Docker Pulls](https://img.shields.io/docker/pulls/oaresearch/optiattack.svg)](https://hub.docker.com/r/oaresearch/optiattack)
![Docker Image Version](https://img.shields.io/docker/v/oaresearch/optiattack)
![Docker Image Size](https://img.shields.io/docker/image-size/oaresearch/optiattack)
[![PyPI Downloads](https://static.pepy.tech/badge/optiattack-client)](https://pepy.tech/projects/optiattack-client)
[![codecov](https://codecov.io/gh/OAResearch/optiattack/graph/badge.svg?token=MOQRTG0OT1)](https://codecov.io/gh/OAResearch/optiattack)

# OptiAttack

OptiAttack (OA) is an optimization-based framework for adversarial example generation designed for network test generation. It provides a modular, extensible platform for generating adversarial examples using evolutionary and search-based algorithms, with a focus on image-based attacks. The project includes both a core engine and a client-server architecture for remote testing, as well as a modern web UI for interactive use.

---

## Table of Contents

- [Features](#features)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Web UI](#web-ui)
- [Client Usage](#client-usage)
- [Baselines](#baselines)
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

## Baselines

OptiAttack can run attacks of the [Adversarial Robustness Toolbox (ART)](https://github.com/Trusted-AI/adversarial-robustness-toolbox)
against the same NUT, through the same HTTP client, so that they can be compared with OptiAttack's own algorithms.

### Installation

ART is an optional dependency:

```bash
pip install -r requirements-baselines.txt
```

Without it, `--algorithm baseline` stops with an error asking you to install `requirements-baselines.txt`.
The other algorithms are not affected. `--pixel_es 0` (CMA-ES) additionally needs the `cma` package.
ART is not installed in the PyPy tox environments, so the baseline tests are skipped on PyPy.
The installed ART version is logged when a baseline starts; report it in papers.

### The NUT must report its labels

Baselines need the full class list, so the NUT has to pass `labels=` (in the model's output order) to `collect_info`:

```python
@collect_info(host="localhost", port=38000, labels=["zebra", "horse", "dog"])
def predict(image_array, additional_data=None):
    return {"predictions": [{"label": "zebra", "score": 0.7}, {"label": "horse", "score": 0.2},
                            {"label": "dog", "score": 0.1}]}
```

Class names in `labels` must be unique (use e.g. WordNet synset IDs if names repeat, as in ImageNet's two
"crane" classes), and a result must not list the same label twice. Scores must be in [0, 1] and sum to at most 1;
labels outside `labels` are an error. Classes the NUT does not
return (e.g. only the top-5) share the remaining probability mass equally.

### Example commands

```bash
python main.py --algorithm baseline --baseline_attack square --max_evaluations 1000   # Square, L-inf
python main.py --algorithm baseline --baseline_attack hsj    --max_evaluations 1000   # HopSkipJump, label-only
python main.py --algorithm baseline --baseline_attack simba  --max_evaluations 1000   # SimBA
python main.py --algorithm baseline --baseline_attack pixel  --max_evaluations 1000 --pixel_th 10   # PixelAttack, L0
```

Per-attack parameters are `square_*`, `hsj_*`, `simba_*` and `pixel_*` (see [docs/parameters.md](docs/parameters.md)).

- **Stop policy.** Score-based attacks (Square, SimBA, PixelAttack) stop at the first success, like OptiAttack's
  algorithms. HopSkipJump uses the whole budget and keeps the successful image with the smallest perturbation.
  `--baseline_stop_at_success auto|true|false` overrides this per run for Square and HopSkipJump. SimBA and
  PixelAttack stop by themselves on a label flip inside ART, so `false` has no effect for them (a warning is logged).
- `--hsj_init_image <path>` gives HopSkipJump an initial adversarial image (e.g. an image of the target class);
  otherwise it starts from a random image drawn with OptiAttack's seeded randomness.
- **Comparison metrics.** Every algorithm writes `first_success_eval` (number of the first successful query, empty if
  none) to `statistics.csv/json`; compare it together with the perturbation (L0/L2/Linf) at an equal budget.
  `config_*` columns (e.g. `config_algorithm`, `config_baseline_attack`) record the settings.

### Fairness

- Same NUT, over HTTP, through `RemoteController.new_action`; the baseline sees only what the NUT returns, the same
  top-k information as OptiAttack's algorithms.
- Every NUT query is counted and the budget (`max_evaluations`, or the time budget) is never exceeded.
- Success uses OptiAttack's fitness formula (`stc.is_solution`), on the image rounded to uint8 that is actually sent.
- Output goes to the usual files (`statistics.csv/json`, final image, overlay, line plot).

### Known limitations

- **Query cache.** Identical images are answered from a cache and are neither re-sent nor counted. This assumes a
  deterministic NUT (ART's Square otherwise spends about 3 queries per iteration on duplicates). The original image
  is answered from OptiAttack's initial NUT call, which is not counted for any algorithm.
- `--square_max_iter 0` means auto: `max_evaluations` for evaluation budgets, 10000 for time budgets, because Square's
  schedule is relative to `max_iter`. In auto mode Square can finish a few queries under the budget when candidate
  images repeat (cache hits do not use budget).
- On very small images Square uses a capped square size (ART workaround). L2 Square rejects portrait images, where
  ART's initialisation breaks; use `--square_norm inf`.
- SimBA uses the pixel basis (`px`) by default. The `dct` basis (ART's default) spreads each step over the whole
  image, so on large images the steps vanish when the image is rounded to uint8 for the NUT unless `simba_epsilon`
  is large. `dct` and `diag` also need square images.
- An attack that only produces images that were already scored (1000 predictions in a row without a new query) is
  stopped with a warning, since it cannot spend the rest of the budget.
- Only queries that improve on the original image become the reported best; if none does, the original prediction
  is reported and the attack counts as failed.
- HopSkipJump may stop before the budget when its own `hsj_max_iter` / `hsj_max_eval` are reached.
- HopSkipJump gets its start image as `x_adv_init`, so ART skips its own initial L2 binary search; the first
  iteration still binary-searches in `hsj_norm`. This is a small difference from stock ART.
- SimBA and PixelAttack stop on a label flip internally, so a non-zero `fitness_threshold` triggers a warning. The
  same warning is given for HopSkipJump, which only sees labels: a negative threshold is practically unreachable.
- ART's SimBA queries both x-εq and x+εq in every iteration (2 queries per iteration), while the SimBA paper queries
  the second only if the first fails. SimBA query counts are therefore higher than in the paper.

### Fair comparison protocol (for papers)

- Report different threat models in separate tables (L0: OptiAttack, PixelAttack; L-inf/L2: Square, SimBA as reference).
- Report HopSkipJump (label-only) in its own information-level category.
- Give both metrics together: success rate and perturbation at equal budget, and queries to first success
  (`first_success_eval`); never give a median over successes without the success rate.
- Show success versus budget curves instead of a single budget point.
- Use default hyperparameters for everyone, or the same tuning budget for everyone, and list the values used.
- State the ART version (it is logged at start); use several seeds and identical images, with Wilcoxon and
  Vargha-Delaney A12 tests.
- State that ART's implementations were used (e.g. ART's SimBA variant, which spends 2 queries per iteration).

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

## Publications  
This project is accompanied by academic research. A list of publications can be found in [`docs/publications.md`](./docs/publications.md).  

To cite our work, please refer to the relevant paper(s) listed there.  

___

## Acknowledgements

- Developed by OptiAttack Team.
- Built with FastAPI, Gradio, and other open-source libraries.

---

**For more details, see the [docs/parameters.md](docs/parameters.md) file and in-code documentation.**
