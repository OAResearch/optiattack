"""Runs every target x method x image x seed as a separate main.py process, with resume and a merged CSV."""
import csv
import hashlib
import json
import logging
import os
import re
import subprocess
import sys
import warnings
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from core.experiment.config import Experiment, Target

logger = logging.getLogger(__name__)

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MAIN_PY = os.path.join(REPO_ROOT, "main.py")
IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp")
PREFIX_COLUMNS = ["target", "method", "image", "seed"]
FAILURE_COLUMNS = PREFIX_COLUMNS + ["returncode", "stderr"]
STDERR_TAIL = 2000


def list_images(folder: str) -> List[Tuple[str, str]]:
    """Return (path relative to folder, absolute path) of every image below folder, sorted by path."""
    found = []
    for root, _dirs, files in os.walk(folder):
        for name in files:
            if name.lower().endswith(IMAGE_EXTENSIONS):
                full = os.path.abspath(os.path.join(root, name))
                found.append((os.path.relpath(full, os.path.abspath(folder)), full))
    return sorted(found, key=lambda item: item[1])


def image_labels(images: List[Tuple[str, str]]) -> List[Tuple[str, str]]:
    """Return (label, absolute path); labels are file stems, made unique by the relative path when stems repeat."""
    def stem(rel):
        return os.path.splitext(os.path.basename(rel))[0]

    def flat(rel, keep_ext):
        rel = rel if keep_ext else os.path.splitext(rel)[0]
        return rel.replace("\\", "__").replace("/", "__")

    counts: Dict[str, int] = {}
    for rel, _ in images:
        counts[stem(rel)] = counts.get(stem(rel), 0) + 1
    labels = [(stem(rel) if counts[stem(rel)] == 1 else flat(rel, False), full) for rel, full in images]
    seen: Dict[str, int] = {}
    for label, _ in labels:
        seen[label] = seen.get(label, 0) + 1
    # Same relative path without extension (a.png and a.jpg side by side): keep the extension.
    return [(label if seen[label] == 1 else flat(rel, True), full)
            for (label, full), (rel, _) in zip(labels, images)]


def _render(value: Any) -> str:
    """Render a parameter value as a command-line string."""
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def _run_params(exp: Experiment, method_name: str) -> Dict[str, Any]:
    """Merge budget, common and method parameters (later overrides earlier)."""
    return {**exp.budget, **exp.common, **exp.methods[method_name]}


def build_command(exp: Experiment, target: Target, method_name: str, image: str, seed: int, out_dir: str,
                  python: str = sys.executable) -> List[str]:
    """Build the main.py command line (a list, never a shell string) for one run."""
    cmd = [python, MAIN_PY,
           "--input_image", image, "--nut_host", target.host, "--nut_port", str(target.port),
           "--seed", str(seed), "--output_dir", out_dir, "--show_plots", "false", "--show_progress", "false",
           "--attack_type", target.attack_type]
    if target.target_class:
        cmd += ["--target", target.target_class]
    for key, value in _run_params(exp, method_name).items():
        if value is not None:
            cmd += [f"--{key}", _render(value)]
    return cmd


def _stats_name(exp: Experiment, method_name: str) -> str:
    """Name of the statistics CSV file of a method's runs."""
    return str(_run_params(exp, method_name).get("statistics_file_name") or "statistics") + ".csv"


def _run_dir(exp: Experiment, target: Target, method: str, label: str, seed: int) -> str:
    """Return the output folder of one run."""
    return os.path.join(os.path.abspath(exp.output), target.name, method, label, f"seed_{seed}")


def _execute(cmd: List[str]) -> Tuple[int, str]:
    """Run one command and return (returncode, stderr tail); the child uses UTF-8 for its files."""
    env = dict(os.environ, PYTHONUTF8="1", PYTHONIOENCODING="utf-8")
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env, check=False)
    return proc.returncode, proc.stderr.decode("utf-8", errors="replace")[-STDERR_TAIL:]


def _read_stats(path: str) -> Optional[Dict[str, str]]:
    """Read the single data row of a statistics CSV (None if unreadable)."""
    try:
        with open(path, newline="", encoding="utf-8", errors="replace") as f:
            return next(csv.DictReader(f), None)
    except OSError:
        return None


def _git(repo: str, *args: str) -> Optional[bytes]:
    """Return the stdout of a git command in repo, or None if git is missing or the command fails."""
    try:
        out = subprocess.run(["git", *args], cwd=repo, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, check=False)
    except OSError:
        return None
    return out.stdout if out.returncode == 0 else None


def _git_info(repo: str) -> Dict[str, Any]:
    """Return git_commit, git_dirty and git_diff_sha256 of repo (all None outside a git repository)."""
    # git_dirty: `git status --porcelain` is non-empty. git_diff_sha256 hashes `git diff HEAD` plus the paths and
    # contents of untracked files (None when clean), so uncommitted and never-committed code is identified too.
    head = _git(repo, "rev-parse", "HEAD")
    status = _git(repo, "status", "--porcelain")
    if head is None or status is None:
        return {"git_commit": None, "git_dirty": None, "git_diff_sha256": None}
    dirty = bool(status.strip())
    digest = None
    if dirty:
        sha = hashlib.sha256(_git(repo, "diff", "HEAD", "--binary") or b"")
        untracked = (_git(repo, "ls-files", "--others", "--exclude-standard", "-z") or b"").split(b"\0")
        for path in sorted(p for p in untracked if p):
            sha.update(b"\0untracked\0" + path + b"\0")
            try:
                with open(os.path.join(repo, os.fsdecode(path)), "rb") as f:
                    sha.update(f.read())
            except OSError:
                pass
        digest = sha.hexdigest()
    return {"git_commit": head.decode("utf-8", errors="replace").strip() or None, "git_dirty": dirty,
            "git_diff_sha256": digest}


def _client_version(repo: str) -> Optional[str]:
    """Return the optiattack_client version from repo's client/setup.py, else the installed one, else None."""
    try:
        with open(os.path.join(repo, "client", "setup.py"), encoding="utf-8") as f:
            match = re.search(r"""version\s*=\s*["']([^"']+)["']""", f.read())
        if match:
            return match.group(1)
    except OSError:
        pass
    try:
        from importlib.metadata import PackageNotFoundError, version
        return version("optiattack_client")
    except PackageNotFoundError:
        return None


def _art_version() -> Optional[str]:
    """Return the ART version if ART is importable."""
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            import art  # type: ignore[import-untyped]
        return str(art.__version__)
    except ImportError:
        return None


def _write_results(exp: Experiment, path: str) -> int:
    """Rebuild results.csv from the statistics files on disk; return the number of rows."""
    rows: List[Dict[str, str]] = []
    columns: List[str] = []
    for target in exp.targets:
        for method in exp.methods:
            for label, _full in image_labels(list_images(target.images)):
                for seed in exp.seeds:
                    stats = _read_stats(os.path.join(_run_dir(exp, target, method, label, seed),
                                                     _stats_name(exp, method)))
                    if not stats:
                        continue
                    row = {"target": target.name, "method": method, "image": label, "seed": str(seed)}
                    row.update({k: v for k, v in stats.items() if k not in PREFIX_COLUMNS})
                    rows.append(row)
                    columns.extend(c for c in row if c not in columns)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=PREFIX_COLUMNS + [c for c in columns if c not in PREFIX_COLUMNS],
                                restval="")
        writer.writeheader()
        writer.writerows(rows)
    return len(rows)


def run_experiment(exp: Experiment, python: str = sys.executable) -> str:
    """Run the whole experiment (resuming finished runs) and return the path of results.csv."""
    out_root = os.path.abspath(exp.output)
    os.makedirs(out_root, exist_ok=True)
    started = datetime.now().isoformat(timespec="seconds")
    failures_path = os.path.join(out_root, "failures.csv")
    if os.path.exists(failures_path):
        os.remove(failures_path)  # unfinished runs are retried, so failures are recorded afresh

    plan = [(t, m, label, full, s) for t in exp.targets for m in exp.methods
            for label, full in image_labels(list_images(t.images)) for s in exp.seeds]
    for index, (target, method, label, image, seed) in enumerate(plan, 1):
        run_dir = _run_dir(exp, target, method, label, seed)
        tag = f"[{index}/{len(plan)}] {target.name}/{method}/{label}/seed_{seed}"
        if os.path.isfile(os.path.join(run_dir, _stats_name(exp, method))):
            logger.info("%s: already finished, skipping", tag)
            continue
        os.makedirs(run_dir, exist_ok=True)
        logger.info("%s: running", tag)
        returncode, stderr = _execute(build_command(exp, target, method, image, seed, run_dir, python))
        if returncode == 0 and os.path.isfile(os.path.join(run_dir, _stats_name(exp, method))):
            continue
        if returncode == 0:
            # main.py calls quit() on startup errors (image or NUT problems), which exits with 0.
            stderr = ("main.py exited without writing statistics (see its log for image/NUT errors)\n"
                      + stderr)[-STDERR_TAIL:]
        logger.error("%s: failed (returncode %s)", tag, returncode)
        new_file = not os.path.exists(failures_path)
        with open(failures_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            if new_file:
                writer.writerow(FAILURE_COLUMNS)
            writer.writerow([target.name, method, label, seed, returncode, stderr])

    results_path = os.path.join(out_root, "results.csv")
    n_rows = _write_results(exp, results_path)
    logger.info("%d of %d runs have results: %s", n_rows, len(plan), results_path)
    info = {"yaml": exp.yaml_text, "python": sys.version, "art_version": _art_version(), **_git_info(REPO_ROOT),
            "optiattack_client_version": _client_version(REPO_ROOT), "start": started,
            "end": datetime.now().isoformat(timespec="seconds")}
    with open(os.path.join(out_root, "run_info.json"), "w", encoding="utf-8") as f:
        json.dump(info, f, indent=2, ensure_ascii=False)
    return results_path
