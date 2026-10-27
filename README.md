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
- [Batch experiments](#batch-experiments)
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

Without it, `--algorithm baseline` (and `--algorithm whitebox`) stops with an error asking you to install
`requirements-baselines.txt`.
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
python main.py --algorithm baseline --baseline_attack sparse_rs --max_evaluations 1000 --sparse_rs_k 10   # Sparse-RS, L0
```

| `--baseline_attack` | Attack | Threat model | NUT output | Implementation |
|---|---|---|---|---|
| `square` | Square Attack | L-inf / L2 | scores | ART |
| `hsj` | HopSkipJump | L2 / L-inf | label only | ART |
| `simba` | SimBA | L2 | scores | ART |
| `pixel` | PixelAttack | L0 (`pixel_th` pixels) | scores | ART |
| `sparse_rs` | Sparse-RS (L0) | L0 (k pixels) | scores | numpy port of the official code |

Per-attack parameters are `square_*`, `hsj_*`, `simba_*`, `pixel_*` and `sparse_rs_*` (see
[docs/parameters.md](docs/parameters.md)).

- **Stop policy.** Score-based attacks (Square, SimBA, PixelAttack) stop at the first success, like OptiAttack's
  algorithms. HopSkipJump uses the whole budget and keeps the successful image with the smallest perturbation.
  `--baseline_stop_at_success auto|true|false` overrides this per run for Square and HopSkipJump. SimBA and
  PixelAttack stop by themselves on a label flip inside ART, so `false` has no effect for them (a warning is logged).
- `--hsj_init_image <path>` gives HopSkipJump an initial adversarial image (e.g. an image of the target class);
  otherwise it starts from a random image drawn with OptiAttack's seeded randomness.
- **Comparison metrics.** Every algorithm writes `first_success_eval` (number of the first successful query, empty if
  none) to `statistics.csv/json`; compare it together with the perturbation (L0/L2/Linf) at an equal budget.
  `config_*` columns (e.g. `config_algorithm`, `config_baseline_attack`) record the settings.
  `gradient_eval_count` is the number of gradient queries (`/gradientNut`) sent to a white-box NUT; it is 0 for
  black-box runs. Each gradient query also counts as one query of `max_evaluations` (shared budget), so
  `eval_count` includes it.

### Perturbation metrics

Every run writes the size of its perturbation to `statistics.csv/json`, computed from the in-memory images (the
original resized image against the adversarial image), so it does not depend on `save_images`:

- `l0_pixels` (pixels with at least one changed channel), `l2`, `linf`, `ssim`, `psnr`: the final solution (the pruned
  one when pruning is enabled).
- `not_minimized_l0_pixels`, `not_minimized_l2`, `not_minimized_linf`, `not_minimized_ssim`, `not_minimized_psnr`:
  the same for the solution before pruning; empty when pruning is disabled.
- All values use the 0-255 scale over all channels. SSIM and PSNR use `data_range=255`; PSNR is `inf` for identical images.
- SSIM needs scikit-image (`pip install -r requirements-experiments.txt`); without it the `ssim` columns stay empty
  and a warning is logged once.
- New columns are added to the CSV with every release, and the `config_*` columns change with every new parameter:
  read the CSV by column name, never by position.

### Pixel budget (`--max_pixels`)

`--max_pixels k` caps how many pixels any image sent to the NUT may change (a pixel counts when at least one channel
differs from the original). The default `0` means no limit.

- Candidates are repaired, not rejected, right before the NUT query (in the fitness function). If the archive image plus
  the candidate would change more than k pixels, randomly chosen candidate actions on not-yet-changed pixels (seeded by
  `--seed`) are moved, with their colour, onto randomly chosen pixels that are already changed (by the archive or by
  the kept new actions) until the image fits. No action is removed, so once k pixels are used the search keeps
  spending its budget by recolouring them. An image above k is never sent.
- A candidate without any action is not queried and costs no budget; it gets the fitness of the original image.
- Works for all OptiAttack algorithms (`random`, `mio`, `genetic`, `abc`, `mabc`) without changes to them.
- Baselines: `square`, `simba` and `hsj`, and the white-box attacks, change the whole image and stop with an error
  when `max_pixels > 0`; `pixel` stops with an error when `pixel_th > max_pixels`, and `sparse_rs` when
  `sparse_rs_k > max_pixels`. These checks run before any NUT query. `sparse_rs` with `--sparse_rs_k 0` (the default) uses `max_pixels` as its k.

### Sparse-RS (`--baseline_attack sparse_rs`)

OptiAttack's main L0 competitor: a numpy port of the L0 (pixel) version of Sparse-RS (Croce, Andriushchenko, Singh,
Flammarion, Hein, AAAI 2022), following `rs_attacks.py` of the [official code](https://github.com/fra31/sparse-rs)
at commit `21d875969a1455e4d5b26dcf32c843e6262d1f9c`. Patch and frame versions are not included. It uses the same
query cache, budget, success formula and stop rule as the ART baselines (ART must be installed, as for every baseline).

- k random pixels get random corner colours ({0, 255} per channel). Every iteration moves
  `max(int(p * k), 1)` of them to new random pixels with new corner colours, and the candidate is kept only if its loss
  is strictly lower. `p` starts at `--sparse_rs_p_init` and shrinks with the official L0 schedule (`p_selection`,
  rescaled to the run length). Every image sent to the NUT changes exactly k pixels.
- `--sparse_rs_k`: k. `0` (default) uses `--max_pixels`, or 10 if that is 0 too.
- `--sparse_rs_p_init`: default 0.3, the official L0 default (`eval.py` and the official README for untargeted
  runs). The official README suggests 0.1 (with 100000 queries) for targeted runs.
- The schedule length is `max_evaluations` (10000 for time budgets), like the official `n_queries`; the run ends when
  the budget is used up or at the first success.
- Seeded by `--seed`; runs are reproducible, but not identical to the official PyTorch runs.

Deviations from the official code (also listed in `core/search/algorithms/baseline/sparse_rs.py`):

- **Loss.** The margin of the NUT's probabilities instead of its logits (untargeted: original-class score minus the best
  other score; targeted: best non-target score minus the target score). The official targeted attack uses the
  cross-entropy loss. This loss orders images like OptiAttack's fitness until the class changes. With
  `--baseline_stop_at_success false` the attack keeps minimising this margin after the first success, while the
  reported best follows the fitness (OptiAttack's untargeted fitness compares the top two scores once the class has
  changed), so the two can disagree after a success.
- **Corner colours.** Every newly changed pixel gets a corner colour different from its original colour, so every
  candidate changes exactly k pixels and differs from the current best. The official code enforces this only for
  1-pixel updates; it only matters for pixels whose original colour is already a corner colour.
- **Stopping.** The official loop stops when its margin becomes negative; here OptiAttack's success formula,
  `fitness_threshold` and `baseline_stop_at_success` decide (the default stops at the first success).
- One image and one restart; no `constant_schedule`.
- Checked against the official code on a toy linear model (32x32, k=10, 200 queries, 300 seeds): success rate 0.80 vs
  0.81 and median queries 124 vs 129 (port vs official). The test `test_sparse_rs_matches_official_distribution` repeats
  this check on 20 seeds when `SPARSE_RS_OFFICIAL` points to a checkout and torch is installed.

### White-box attacks (`--algorithm whitebox`)

`--algorithm whitebox --whitebox_attack fgsm|pgd|cw` runs ART's FGSM, PGD or Carlini-Wagner L2 with the NUT's own
gradients. It is a separate algorithm name (not a `baseline_attack`), so black-box and white-box runs never mix in
result files; it shares the baselines' query tracker, cache, archive and warnings. It needs ART
(`requirements-baselines.txt`) and a NUT that passes `gradient_fn` to `collect_info` (client 0.2.0, see
[client/README.md](client/README.md)) and returns `logits` (all classes, in `labels` order) next to its `predictions`.

```bash
python main.py --algorithm whitebox --whitebox_attack pgd  --max_evaluations 1000   # PGD, L-inf (default)
python main.py --algorithm whitebox --whitebox_attack fgsm --max_evaluations 1000   # FGSM, L-inf, one gradient
python main.py --algorithm whitebox --whitebox_attack cw   --max_evaluations 1000   # Carlini-Wagner L2
```

| `--whitebox_attack` | Attack (ART) | Norm | Gradient used | Parameters |
|---|---|---|---|---|
| `fgsm` | `FastGradientMethod` | L-inf | `loss` (cross-entropy) | `fgsm_eps` |
| `pgd` | `ProjectedGradientDescent` | `pgd_norm` (`inf` or `2`) | `loss` (cross-entropy) | `pgd_*` |
| `cw` | `CarliniL2Method` | L2 | `class` (logits) | `cw_*` |

- **Checks before any query.** The NUT must declare `"gradients"`, and the result of OptiAttack's initial `run_nut`
  call must carry `logits`; otherwise the run stops with an error that says what is missing. `--max_pixels > 0` is
  rejected like for the dense baselines, since these attacks change the whole image.
- **Shared budget.** Every gradient query (`/gradientNut`) counts as one query of `max_evaluations`, like a prediction,
  and is also counted in `gradient_eval_count`. The attack may use `max_evaluations - 1` queries; the last one is kept
  for the final check.
- **Final check.** The attack's result is rounded to uint8 and predicted with a normal prediction query (answered from
  the cache if the attack already predicted that image, e.g. PGD's own success check). If the budget stops the attack
  first, the image it was last working on is checked instead. Success uses the Phase 1 formula on these uint8
  predictions (`fitness_from_result` + `stc.is_solution`); logits are never used for success.
- **Gradients are white-box only.** Only `algorithm: whitebox` builds an estimator that may query gradients; the
  black-box baselines' estimator raises `RuntimeError` on any gradient call, so they can never use them.
- **Reported image.** As for the baselines, the query tracker reports the best predicted image (success first, then
  the smallest perturbation in the attack's norm). Every prediction the attack makes is tracked, e.g. C&W's line-search
  images. White-box attacks do not stop at the first success: they run their configured iterations
  (`baseline_stop_at_success` does not apply).
- **Logits.** In white-box mode the estimator's `predict` returns the NUT's logits, so C&W's loss and its `class`
  gradients come from the same quantity. Predictions are made on the uint8-rounded image, gradients at the float32
  image.
- **Cache.** Gradients are cached by (float32 image bytes, label, kind); repeated requests (e.g. a PGD iterate that has
  stopped moving at the corner of the L-inf ball) cost no query.
- **Budget warning.** If `pgd_max_iter` (one gradient per iteration) or `2 * cw_binary_search_steps * cw_max_iter`
  (two gradients per C&W iteration) exceeds the budget left for the attack, a warning is logged before it starts.

**Defaults and measured query costs.** Measured on a 224x224 linear fake NUT (`logits = W @ x + b`, analytic
gradients, 6 classes) with `max_evaluations 1000` (script in the task report; numbers are queries sent to the NUT):

- `fgsm_eps 8`, `pgd_eps 8`, `pgd_eps_step 2`: 8/255 and 2/255 on the 0-255 scale. FGSM costs 1 gradient + 1 prediction.
- `pgd_max_iter 100` (ART default): at most one gradient query per iteration, plus one prediction for the result.
  L-inf PGD used 58 gradient queries for 100 iterations (the rest were cache hits once the iterate stopped moving) and
  59 queries in total; L2 PGD used 100 + 1. All iterations always fit in 1000 queries.
- C&W uses ART's defaults for `cw_max_iter 10`, `cw_binary_search_steps 10`, `cw_max_halving 5`, `cw_max_doubling 5`
  and `cw_confidence 0`. One iteration costs 2 gradient queries plus up to `cw_max_halving + cw_max_doubling + 1`
  (= 11) line-search predictions, so at most 13 queries per iteration and 1 + 10 * 13 = 131 per binary-search step:
  at least one full binary-search step (in practice all 10) fits in 1000 queries. Measured: 167 queries for all 100
  iterations (98 gradient, 69 prediction queries, 1.7 per iteration; repeated images are cache hits); the first
  binary-search step finished within 18 queries. On 32x32 and 224x224 fakes (5 models each, untargeted and targeted)
  every run succeeded and used 62-173 queries.
- `cw_initial_const 650.25` is ART's 0.01 rescaled to 0-255 (the L2 term grows by 255^2). `cw_learning_rate 0.01`
  keeps ART's value: the rescaled rate (0.01 / 255^2) makes steps that vanish when the image is rounded to uint8 for
  the NUT (measured: C&W changed no pixel), while 0.01 succeeded in every run above.
- These defaults fit the 1000-query budget; for other budgets or models, tune them and report the values used.

### Gradient oracle (`--use_gradients`)

`GradientOracle` (`core/search/service/gradient_oracle.py`, container provider `gradient_oracle`) is a hook for future
gradient-guided OptiAttack algorithms; no current algorithm uses it. `available()` is True only if `--use_gradients true`
and the NUT declares `"gradients"`. `pixel_saliency(image)` returns an HxW map (L2 norm over channels of the `loss`
gradient of the original class) and counts as one gradient query in the shared budget. `--use_gradients` defaults to
`false` and must stay off for black-box results, so black-box comparisons remain gradient-free.

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

- Report different threat models in separate tables (L0: OptiAttack, Sparse-RS, PixelAttack; L-inf/L2: Square, SimBA as
  reference). Use the same k for OptiAttack (`--max_pixels`) and Sparse-RS.
- Report HopSkipJump (label-only) in its own information-level category.
- Give both metrics together: success rate and perturbation at equal budget, and queries to first success
  (`first_success_eval`); never give a median over successes without the success rate.
- Show success versus budget curves instead of a single budget point.
- Use default hyperparameters for everyone, or the same tuning budget for everyone, and list the values used.
- State the ART version (it is logged at start); use several seeds and identical images, with Wilcoxon and
  Vargha-Delaney A12 tests.
- State that ART's implementations were used (e.g. ART's SimBA variant, which spends 2 queries per iteration).

---

## Batch experiments

`python -m core.experiment run experiment.yaml` runs every target x method x image x seed as a separate `main.py`
process (so no state leaks between runs) and merges the results. It works with your own remote NUT and image
folder, and needs `pip install -r requirements-experiments.txt` (pyyaml, scipy, scikit-image). See the commented
[examples/experiment.yaml](examples/experiment.yaml).

- **Targets.** A top-level `images` + `nut: {host, port}` (+ `attack_type`, `target`) is one target named `default`;
  or give a `targets:` list of `{name, images, nut, attack_type?, target?}` to use several NUTs. Relative `images`
  and `output` paths are resolved against the folder of the YAML file, not the current directory.
- **Parameters.** `budget`, `common` and each entry of `methods` hold `main.py` parameters (see
  `docs/parameters.md`); method values override `common`, which overrides `budget`. Unknown names are rejected, as
  are the ones the runner sets itself (`input_image`, `nut_host`, `nut_port`, `seed`, `output_dir`, `show_plots`,
  `show_progress`, `attack_type`, `target`).
- **Pruning is per method.** Enable `enable_pruning` only on L0 methods; the validator warns if it is enabled for
  the dense baselines `square`, `simba` or `hsj`, or for a white-box method (`algorithm: whitebox`).
- **Output.** Each run writes to `<output>/<target>/<method>/<image>/seed_<n>/`. `results.csv` has every finished
  run's statistics row prefixed with `target, method, image, seed`; `run_info.json` stores the YAML, Python and ART
  versions, the git commit, `git_dirty` (uncommitted changes, including untracked files), `git_diff_sha256` (a
  hash of `git diff HEAD` plus the untracked files; null when clean), `optiattack_client_version` and the start and
  end times. Images are found recursively (`*.png, *.jpg, *.jpeg,
  *.bmp`); if two images share a file name, their relative path (with `__` for separators) names them.
- **Resume and failures.** A run whose folder already has `statistics.csv` is skipped, so an interrupted experiment
  continues when you run the same command again. Failed runs (non-zero exit, or `main.py` stopping without writing
  statistics, e.g. an unreachable NUT) are listed in `failures.csv` and retried on the next run; the experiment goes
  on.
- **Summary and statistics.** `run` ends by writing `summary.md`, `summary.csv` and `pairwise.csv` into the output
  folder; `python -m core.experiment summarize results.csv [--yaml experiment.yaml]` recomputes them next to
  `results.csv` (without the YAML: k values 10/20/50/100, no threat-model groups).
  `summary.csv` has one row per (target, method): `n_runs`, `success_rate`, `queries_median/q1/q3` (runs that
  succeeded, from `first_success_eval`), medians of `l0_pixels`, `l2`, `linf`, `ssim`, `psnr` (successful runs),
  the median `pruning_eval_count`, and `success@k` for each `analysis.k_values` (fraction of all runs that
  succeeded with `l0_pixels <= k`). Cells stay blank where no run succeeded.
- **Paired comparisons.** Within each `analysis.threat_models` group (one Markdown table per group; a listed method
  without runs is shown as "no runs"), every method pair is matched on (target, image, seed) and compared with:
  the paired Wilcoxon signed-rank test and the Vargha-Delaney A12 on queries and on L0, and the exact McNemar test
  on success. A failed run counts as its own `max_evaluations + 1` queries, read from the run's
  `config_max_evaluations` column (so a budget set in `budget`, `common`, per method or left at the default is
  always right); a pair whose runs used a time budget (`config_stopping_criterion: time`) reports its query test
  as "n/a (time budget)", and a pair whose two methods had different budgets gets a note in `summary.md`. L0 is compared only on pairs where both runs succeeded. A12 is
  shown as 1 - A12, the probability that the first method is better (fewer queries, lower L0). Holm-adjusted
  p-values are computed per threat-model group, separately for each test family (queries, L0, success).

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
