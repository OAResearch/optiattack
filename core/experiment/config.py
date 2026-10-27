"""Loading and validation of the batch experiment YAML file."""
import logging
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from core.config_parser import ConfigParser

logger = logging.getLogger(__name__)

# Parameters that the runner sets itself for every run; a method or `common` must not set them.
RUNNER_OWNED_KEYS = frozenset({"input_image", "nut_host", "nut_port", "seed", "output_dir", "show_plots",
                               "show_progress", "attack_type", "target"})
# Baselines that perturb (almost) every pixel; pruning them is very expensive and makes L0 meaningless.
DENSE_BASELINE_ATTACKS = frozenset({"square", "simba", "hsj"})


class ExperimentConfigError(ValueError):

    """The experiment file is invalid."""


@dataclass
class Target:

    """One network under test together with the images to attack on it."""

    name: str
    images: str
    host: str
    port: int
    attack_type: str
    target_class: Optional[str]


@dataclass
class Experiment:

    """A validated experiment description."""

    name: str
    targets: List[Target]
    seeds: List[int]
    budget: Dict[str, Any]
    output: str
    common: Dict[str, Any]
    methods: Dict[str, Dict[str, Any]]
    analysis: Dict[str, Any]
    yaml_text: str = field(default="")


def _load_yaml(text: str) -> Any:
    """Parse YAML text, importing pyyaml lazily."""
    try:
        import yaml  # type: ignore[import-untyped,unused-ignore]
    except ImportError as e:
        raise ImportError("Batch experiments need pyyaml. "
                          "Install it with: pip install -r requirements-experiments.txt") from e
    return yaml.safe_load(text)


def _as_dict(value: Any, what: str) -> Dict[str, Any]:
    """Return value as a dict (None becomes empty) or raise."""
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ExperimentConfigError(f"'{what}' must be a mapping, got {type(value).__name__}")
    return value


def _check_keys(params: Dict[str, Any], owner: str, valid: Any) -> None:
    """Check that every key is a main.py parameter that the runner does not set itself."""
    for key in params:
        if key in RUNNER_OWNED_KEYS:
            raise ExperimentConfigError(
                f"Parameter '{key}' in {owner} is set by the runner itself and cannot be configured here")
        if key not in valid:
            raise ExperimentConfigError(f"Unknown parameter '{key}' in {owner}: it is not a main.py parameter "
                                        f"(see docs/parameters.md)")


def _check_seeds(seeds: Any) -> List[int]:
    """Validate the seed list."""
    if (not isinstance(seeds, list) or not seeds
            or any(isinstance(s, bool) or not isinstance(s, int) for s in seeds)):
        raise ExperimentConfigError("'seeds' must be a non-empty list of integers, e.g. [1, 2, 3]")
    return list(seeds)


def _parse_target(raw: Dict[str, Any], default_name: str, base: str) -> Target:
    """Build one Target from a mapping; a relative images folder is resolved against base."""
    nut = _as_dict(raw.get("nut"), "nut")
    images = raw.get("images")
    name = str(raw.get("name", default_name))
    if not images:
        raise ExperimentConfigError(f"Target '{name}' has no 'images' folder")
    images = _resolve(images, base)
    if "host" not in nut or "port" not in nut:
        raise ExperimentConfigError(f"Target '{name}' needs nut: {{host: ..., port: ...}}")
    try:
        port = int(nut["port"])
    except (TypeError, ValueError):
        raise ExperimentConfigError(f"Target '{name}': nut port must be an integer, got {nut['port']!r}")
    attack_type = str(raw.get("attack_type", ConfigParser.AttackType.UNTARGETED))
    if attack_type not in (ConfigParser.AttackType.TARGETED, ConfigParser.AttackType.UNTARGETED):
        raise ExperimentConfigError(f"Target '{name}': attack_type must be 'targeted' or 'untargeted'")
    target_class = raw.get("target")
    if attack_type == ConfigParser.AttackType.TARGETED and not target_class:
        raise ExperimentConfigError(f"Target '{name}': a targeted attack needs 'target: <class>'")
    if not os.path.isdir(str(images)):
        raise ExperimentConfigError(f"Image folder not found for target '{name}': {images}")
    from core.experiment.runner import list_images
    if not list_images(str(images)):
        raise ExperimentConfigError(f"Image folder for target '{name}' contains no images "
                                    f"(*.png, *.jpg, *.jpeg, *.bmp): {images}")
    return Target(name=name, images=str(images), host=str(nut["host"]), port=port, attack_type=attack_type,
                  target_class=None if target_class is None else str(target_class))


def _dense_attack(merged: Dict[str, Any], defaults: Dict[str, Any]) -> Optional[str]:
    """Return the name of the dense attack a method runs (dense baseline or any white-box attack), else None."""
    algorithm = merged.get("algorithm")
    if algorithm == ConfigParser.Algorithms.BASELINE and merged.get("baseline_attack") in DENSE_BASELINE_ATTACKS:
        return str(merged.get("baseline_attack"))
    if algorithm == ConfigParser.Algorithms.WHITEBOX:
        return f"white-box {merged.get('whitebox_attack') or defaults.get('whitebox_attack')}"
    return None


def _warn_dense_pruning(exp_budget: Dict[str, Any], common: Dict[str, Any],
                        methods: Dict[str, Dict[str, Any]], defaults: Dict[str, Any]) -> None:
    """Warn when pruning is enabled for a dense method (square, simba, hsj, or a white-box attack)."""
    for name, params in methods.items():
        merged = {**exp_budget, **common, **params}
        attack = _dense_attack(merged, defaults)
        if attack is not None and merged.get("enable_pruning") is True:
            logger.warning("Method '%s': enable_pruning is true for the dense attack '%s'. Pruning costs tens of "
                           "thousands of queries per run and makes its L0 meaningless; consider disabling it.",
                           name, attack)


def _resolve(path: Any, base: str) -> str:
    """Return path as a string, joined to base when it is relative."""
    return os.path.normpath(os.path.join(base, str(path)))


def load_experiment(path: str) -> Experiment:
    """Load and validate an experiment YAML file; relative images and output paths are taken from its folder."""
    with open(path, encoding="utf-8") as f:
        text = f.read()
    data = _load_yaml(text)
    if not isinstance(data, dict):
        raise ExperimentConfigError("The experiment file must contain a YAML mapping at the top level")

    valid = ConfigParser().default_params()
    base = os.path.dirname(os.path.abspath(path))  # relative images/output paths are relative to the YAML file
    budget = _as_dict(data.get("budget"), "budget")
    common = _as_dict(data.get("common"), "common")
    _check_keys(budget, "budget", valid)
    _check_keys(common, "common", valid)

    raw_methods = _as_dict(data.get("methods"), "methods")
    if not raw_methods:
        raise ExperimentConfigError("'methods' must define at least one method")
    methods = {}
    for mname, params in raw_methods.items():
        methods[str(mname)] = _as_dict(params, f"methods.{mname}")
        _check_keys(methods[str(mname)], f"method '{mname}'", valid)

    if "targets" in data:
        raw_targets = data["targets"]
        if not isinstance(raw_targets, list) or not raw_targets:
            raise ExperimentConfigError("'targets' must be a non-empty list")
        targets = [_parse_target(_as_dict(t, "targets[]"), f"target{i}", base) for i, t in enumerate(raw_targets)]
        if len({t.name for t in targets}) != len(targets):
            raise ExperimentConfigError("Target names must be unique")
    else:
        targets = [_parse_target({k: v for k, v in data.items() if k != "name"}, "default", base)]

    if not data.get("output"):
        raise ExperimentConfigError("'output' (the output folder) is required")

    _warn_dense_pruning(budget, common, methods, valid)
    name = str(data.get("name") or os.path.splitext(os.path.basename(path))[0])
    return Experiment(name=name, targets=targets, seeds=_check_seeds(data.get("seeds")), budget=budget,
                      output=_resolve(data["output"], base), common=common, methods=methods,
                      analysis=_as_dict(data.get("analysis"), "analysis"), yaml_text=text)
