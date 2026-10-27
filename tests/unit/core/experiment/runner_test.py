"""Tests for the batch experiment runner (uses a fake main.py)."""
import csv
import json
import os
import sys

import pytest

yaml = pytest.importorskip("yaml")

from core.experiment import runner  # noqa: E402
from core.experiment.config import Experiment, Target  # noqa: E402

FAKE_MAIN = '''
import csv, os, sys
a = sys.argv[1:]
d = {a[i][2:]: a[i + 1] for i in range(0, len(a) - 1, 2) if a[i].startswith("--")}
marker = os.environ.get("FAKE_MARKER")
if marker:
    with open(marker, "a", encoding="utf-8") as f:
        f.write(d["input_image"] + "\\n")
if d.get("algorithm") == "crash":
    sys.stderr.write("boom \\u00e7\\n")
    sys.exit(1)
if d.get("algorithm") == "silent":
    sys.exit(0)
os.makedirs(d["output_dir"], exist_ok=True)
with open(os.path.join(d["output_dir"], "statistics.csv"), "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["flipped", "eval_count", "l0_pixels"])
    w.writerow(["True", int(d["seed"]) * 10, len(os.path.basename(d["input_image"]))])
'''


@pytest.fixture
def fake(tmp_path, monkeypatch):
    script = tmp_path / "fake_main.py"
    script.write_text(FAKE_MAIN, encoding="utf-8")
    monkeypatch.setattr(runner, "MAIN_PY", str(script))
    marker = tmp_path / "marker.txt"
    monkeypatch.setenv("FAKE_MARKER", str(marker))
    return marker


def _invocations(marker):
    return len(marker.read_text(encoding="utf-8").split("\n")) - 1


def _exp(tmp_path, images=("a.png", "b.png"), methods=None, seeds=(1, 2), folder="imgs", out="out", **kw):
    d = tmp_path / folder
    d.mkdir(parents=True, exist_ok=True)
    for n in images:
        p = d / n
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(b"x")
    methods = methods if methods is not None else {"m1": {"algorithm": "mio"}, "m2": {"algorithm": "random"}}
    return Experiment(name="t", targets=[Target("tg", str(d), "localhost", 38000, "untargeted", None)],
                      seeds=list(seeds), budget={"max_evaluations": 5}, output=str(tmp_path / out),
                      common=kw.get("common", {}), methods=methods, analysis={})


def _rows(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def test_build_command_precedence(tmp_path):
    exp = _exp(tmp_path, common={"max_evaluations": 7, "save_images": True, "enable_pruning": False},
               methods={"m": {"enable_pruning": True, "algorithm": "mio"}})
    exp.targets[0] = Target("tg", exp.targets[0].images, "hh", 99, "targeted", "cls")
    cmd = runner.build_command(exp, exp.targets[0], "m", "img.png", 3, "outdir", python="PY")
    assert cmd[:2] == ["PY", runner.MAIN_PY]
    pairs = dict(zip(cmd[2::2], cmd[3::2]))
    assert pairs["--input_image"] == "img.png" and pairs["--nut_host"] == "hh" and pairs["--nut_port"] == "99"
    assert pairs["--seed"] == "3" and pairs["--output_dir"] == "outdir"
    assert pairs["--show_plots"] == "false" and pairs["--show_progress"] == "false"
    assert pairs["--attack_type"] == "targeted" and pairs["--target"] == "cls"
    assert pairs["--max_evaluations"] == "7"       # common overrides budget
    assert pairs["--enable_pruning"] == "true"     # method overrides common
    assert pairs["--save_images"] == "true"
    assert cmd.count("--enable_pruning") == 1


def test_build_command_untargeted_has_no_target(tmp_path):
    exp = _exp(tmp_path)
    cmd = runner.build_command(exp, exp.targets[0], "m1", "i.png", 1, "o", python="PY")
    assert "--target" not in cmd


def test_runs_every_combination(tmp_path, fake):
    exp = _exp(tmp_path)
    runner.run_experiment(exp)
    for m in ("m1", "m2"):
        for stem in ("a", "b"):
            for s in (1, 2):
                assert os.path.isfile(os.path.join(exp.output, "tg", m, stem, f"seed_{s}", "statistics.csv"))
    assert _invocations(fake) == 8


def test_resume_skips_finished_runs(tmp_path, fake):
    exp = _exp(tmp_path)
    runner.run_experiment(exp)
    runner.run_experiment(exp)
    assert _invocations(fake) == 8
    os.remove(os.path.join(exp.output, "tg", "m1", "a", "seed_1", "statistics.csv"))
    res = runner.run_experiment(exp)
    assert _invocations(fake) == 9
    assert len(_rows(res)) == 8


def test_failed_run_is_recorded_and_others_continue(tmp_path, fake):
    exp = _exp(tmp_path, methods={"ok": {"algorithm": "mio"}, "bad": {"algorithm": "crash"}})
    res = runner.run_experiment(exp)
    rows = _rows(res)
    assert len(rows) == 4 and {r["method"] for r in rows} == {"ok"}
    fails = _rows(os.path.join(exp.output, "failures.csv"))
    assert len(fails) == 4
    assert {f["method"] for f in fails} == {"bad"}
    assert fails[0]["returncode"] == "1" and "boom" in fails[0]["stderr"]
    assert list(fails[0]) == ["target", "method", "image", "seed", "returncode", "stderr"]


def test_exit_zero_without_statistics_is_a_failure(tmp_path, fake):
    exp = _exp(tmp_path, methods={"s": {"algorithm": "silent"}}, seeds=(1,), images=("a.png",))
    res = runner.run_experiment(exp)
    assert _rows(res) == []
    fails = _rows(os.path.join(exp.output, "failures.csv"))
    assert len(fails) == 1 and fails[0]["returncode"] == "0" and "without writing statistics" in fails[0]["stderr"]


def test_failures_csv_is_rebuilt_not_duplicated_on_rerun(tmp_path, fake):
    exp = _exp(tmp_path, methods={"bad": {"algorithm": "crash"}}, seeds=(1,), images=("a.png",))
    runner.run_experiment(exp)
    runner.run_experiment(exp)
    assert len(_rows(os.path.join(exp.output, "failures.csv"))) == 1


def test_results_csv_merges_all_runs(tmp_path, fake):
    exp = _exp(tmp_path)
    res = runner.run_experiment(exp)
    assert res == os.path.join(exp.output, "results.csv")
    rows = _rows(res)
    assert len(rows) == 8
    with open(res, newline="", encoding="utf-8") as f:
        header = next(csv.reader(f))
    assert header == ["target", "method", "image", "seed", "flipped", "eval_count", "l0_pixels"]
    r = [r for r in rows if r["method"] == "m2" and r["image"] == "b" and r["seed"] == "2"][0]
    assert r["target"] == "tg" and r["eval_count"] == "20"


def test_run_info_written(tmp_path, fake):
    exp = _exp(tmp_path)
    exp.yaml_text = "name: t\n"
    runner.run_experiment(exp)
    with open(os.path.join(exp.output, "run_info.json"), encoding="utf-8") as f:
        info = json.load(f)
    assert info["yaml"] == "name: t\n"
    assert info["python"].startswith(sys.version.split()[0])
    for key in ("start", "end", "git_commit", "git_dirty", "git_diff_sha256", "art_version",
                "optiattack_client_version"):
        assert key in info


def test_runner_handles_spaces_and_non_ascii_paths(tmp_path, fake):
    exp = _exp(tmp_path, images=("görüntü 1.png",), folder="deney ç", out="çıktı klasörü", seeds=(1,),
               methods={"m1": {"algorithm": "mio"}})
    res = runner.run_experiment(exp)
    assert len(_rows(res)) == 1
    assert _rows(res)[0]["image"] == "görüntü 1"
    assert _invocations(fake) == 1
    runner.run_experiment(exp)
    assert _invocations(fake) == 1
    assert len(_rows(res)) == 1


def test_duplicate_image_stems_are_disambiguated(tmp_path, fake):
    exp = _exp(tmp_path, images=("x/a.png", "y/a.png", "b.png"), methods={"m1": {"algorithm": "mio"}}, seeds=(1,))
    res = runner.run_experiment(exp)
    assert sorted(r["image"] for r in _rows(res)) == ["b", "x__a", "y__a"]


def test_list_images_recursive_sorted_and_filtered(tmp_path):
    exp = _exp(tmp_path, images=("b.PNG", "a.jpg", "sub/c.jpeg", "d.bmp", "e.txt"))
    found = runner.list_images(exp.targets[0].images)
    paths = [p for _, p in found]
    assert paths == sorted(paths)
    assert sorted(os.path.basename(p) for p in paths) == ["a.jpg", "b.PNG", "c.jpeg", "d.bmp"]


def _git(repo, *args):
    import subprocess
    subprocess.run(["git", "-c", "user.name=t", "-c", "user.email=t@t", "-c", "commit.gpgsign=false", *args],
                   cwd=str(repo), check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)


@pytest.fixture
def git_repo(tmp_path):
    import shutil
    if shutil.which("git") is None:
        pytest.skip("git is not installed")
    repo = tmp_path / "repo"
    (repo / "client").mkdir(parents=True)
    (repo / "client" / "setup.py").write_text('setup(\n    name="optiattack_client",\n    version="9.8.7",\n)\n',
                                               encoding="utf-8")
    (repo / "code.py").write_text("x = 1\n", encoding="utf-8")
    _git(repo, "init", "-q")
    _git(repo, "add", ".")
    _git(repo, "commit", "-q", "-m", "init")
    return repo


def test_git_info_clean_and_dirty(git_repo):
    clean = runner._git_info(str(git_repo))
    assert len(clean["git_commit"]) == 40
    assert clean["git_dirty"] is False and clean["git_diff_sha256"] is None
    (git_repo / "code.py").write_text("x = 2\n", encoding="utf-8")
    dirty = runner._git_info(str(git_repo))
    assert dirty["git_commit"] == clean["git_commit"]
    assert dirty["git_dirty"] is True and len(dirty["git_diff_sha256"]) == 64
    (git_repo / "code.py").write_text("x = 3\n", encoding="utf-8")
    assert runner._git_info(str(git_repo))["git_diff_sha256"] != dirty["git_diff_sha256"]


def test_git_info_outside_a_repository(tmp_path):
    info = runner._git_info(str(tmp_path))
    assert info == {"git_commit": None, "git_dirty": None, "git_diff_sha256": None}


def test_run_info_identifies_dirty_code_and_client_version(tmp_path, fake, git_repo, monkeypatch):
    (git_repo / "code.py").write_text("x = 2\n", encoding="utf-8")
    monkeypatch.setattr(runner, "REPO_ROOT", str(git_repo))
    exp = _exp(tmp_path)
    runner.run_experiment(exp)
    with open(os.path.join(exp.output, "run_info.json"), encoding="utf-8") as f:
        info = json.load(f)
    assert info["git_dirty"] is True and len(info["git_diff_sha256"]) == 64
    assert info["optiattack_client_version"] == "9.8.7"


def test_client_version_of_this_repository():
    assert runner._client_version(runner.REPO_ROOT) == "0.2.0"


def test_git_info_counts_untracked_files(git_repo):
    # new, never-committed code must also change the fingerprint (git diff HEAD alone misses it)
    clean = runner._git_info(str(git_repo))
    (git_repo / "new_module.py").write_text("y = 1\n", encoding="utf-8")
    first = runner._git_info(str(git_repo))
    assert first["git_dirty"] is True and first["git_diff_sha256"] not in (None, clean["git_diff_sha256"])
    (git_repo / "new_module.py").write_text("y = 2\n", encoding="utf-8")
    assert runner._git_info(str(git_repo))["git_diff_sha256"] != first["git_diff_sha256"]
