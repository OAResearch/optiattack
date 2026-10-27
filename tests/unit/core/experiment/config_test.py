"""Tests for the experiment YAML configuration."""
import logging

import pytest

yaml = pytest.importorskip("yaml")

from core.experiment.config import ExperimentConfigError, load_experiment  # noqa: E402


def _write(tmp_path, data, name="exp.yaml"):
    path = tmp_path / name
    path.write_text(yaml.safe_dump(data, allow_unicode=True), encoding="utf-8")
    return str(path)


def _images(tmp_path, folder="imgs", names=("a.png",)):
    d = tmp_path / folder
    d.mkdir(exist_ok=True)
    for n in names:
        (d / n).write_bytes(b"x")
    return str(d)


def _base(tmp_path, **extra):
    data = {"name": "e", "images": _images(tmp_path), "nut": {"host": "h", "port": 1},
            "seeds": [1, 2], "output": str(tmp_path / "out"),
            "methods": {"mio": {"algorithm": "mio"}}}
    data.update(extra)
    return data


def test_shorthand_yaml_becomes_single_target(tmp_path):
    data = _base(tmp_path, budget={"max_evaluations": 5}, common={"save_images": True})
    exp = load_experiment(_write(tmp_path, data))
    assert exp.name == "e"
    assert len(exp.targets) == 1
    t = exp.targets[0]
    assert (t.name, t.host, t.port, t.attack_type, t.target_class) == ("default", "h", 1, "untargeted", None)
    assert exp.seeds == [1, 2]
    assert exp.budget == {"max_evaluations": 5}
    assert exp.common == {"save_images": True}
    assert exp.methods == {"mio": {"algorithm": "mio"}}
    assert exp.analysis == {}
    assert "name: e" in exp.yaml_text


def test_targets_list(tmp_path):
    data = _base(tmp_path)
    for k in ("images", "nut"):
        del data[k]
    data["targets"] = [
        {"name": "a", "images": _images(tmp_path, "ia"), "nut": {"host": "x", "port": 2}},
        {"name": "b", "images": _images(tmp_path, "ib"), "nut": {"host": "y", "port": 3},
         "attack_type": "targeted", "target": "cls"},
    ]
    exp = load_experiment(_write(tmp_path, data))
    assert [t.name for t in exp.targets] == ["a", "b"]
    assert exp.targets[1].attack_type == "targeted" and exp.targets[1].target_class == "cls"
    assert exp.targets[0].port == 2


def test_unknown_method_key_is_rejected(tmp_path):
    data = _base(tmp_path, methods={"weird": {"algorithm": "mio", "no_such_param": 3}})
    with pytest.raises(ExperimentConfigError) as e:
        load_experiment(_write(tmp_path, data))
    assert "no_such_param" in str(e.value) and "weird" in str(e.value)


def test_runner_owned_key_is_rejected(tmp_path):
    with pytest.raises(ExperimentConfigError, match="seed"):
        load_experiment(_write(tmp_path, _base(tmp_path, methods={"m": {"seed": 3}})))
    with pytest.raises(ExperimentConfigError, match="output_dir"):
        load_experiment(_write(tmp_path, _base(tmp_path, common={"output_dir": "x"})))


def test_empty_image_folder_is_rejected(tmp_path):
    data = _base(tmp_path, images=_images(tmp_path, "empty", names=()))
    with pytest.raises(ExperimentConfigError, match="no images"):
        load_experiment(_write(tmp_path, data))


def test_missing_image_folder_is_rejected(tmp_path):
    with pytest.raises(ExperimentConfigError, match="not found"):
        load_experiment(_write(tmp_path, _base(tmp_path, images=str(tmp_path / "nope"))))


@pytest.mark.parametrize("seeds", [[], "1", [1, "a"], [True], None])
def test_bad_seeds_are_rejected(tmp_path, seeds):
    with pytest.raises(ExperimentConfigError, match="seeds"):
        load_experiment(_write(tmp_path, _base(tmp_path, seeds=seeds)))


def test_targeted_without_target_is_rejected(tmp_path):
    with pytest.raises(ExperimentConfigError, match="target"):
        load_experiment(_write(tmp_path, _base(tmp_path, attack_type="targeted")))


def test_pruning_on_dense_baseline_warns(tmp_path, caplog):
    methods = {"sq": {"algorithm": "baseline", "baseline_attack": "square", "enable_pruning": True},
               "ok": {"algorithm": "mio", "enable_pruning": True}}
    with caplog.at_level(logging.WARNING):
        load_experiment(_write(tmp_path, _base(tmp_path, methods=methods)))
    msgs = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
    assert any("'sq'" in m and "enable_pruning" in m for m in msgs)
    assert not any("'ok'" in m for m in msgs)


def test_pruning_in_common_warns_for_dense_method(tmp_path, caplog):
    methods = {"h": {"algorithm": "baseline", "baseline_attack": "hsj"}}
    with caplog.at_level(logging.WARNING):
        load_experiment(_write(tmp_path, _base(tmp_path, methods=methods, common={"enable_pruning": True})))
    assert any("'h'" in r.getMessage() for r in caplog.records)


def test_missing_methods_is_rejected(tmp_path):
    with pytest.raises(ExperimentConfigError, match="methods"):
        load_experiment(_write(tmp_path, _base(tmp_path, methods={})))


def test_pruning_on_whitebox_method_warns(tmp_path, caplog):
    methods = {"wb": {"algorithm": "whitebox", "whitebox_attack": "fgsm", "enable_pruning": True},
               "wb_plain": {"algorithm": "whitebox"}}
    with caplog.at_level(logging.WARNING):
        load_experiment(_write(tmp_path, _base(tmp_path, methods=methods)))
    msgs = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
    assert any("'wb'" in m and "enable_pruning" in m and "fgsm" in m for m in msgs)
    assert not any("'wb_plain'" in m for m in msgs)


def test_relative_paths_resolve_against_the_yaml_folder(tmp_path, monkeypatch):
    exp_dir = tmp_path / "exp dir"
    exp_dir.mkdir()
    _images(exp_dir, folder="imgs")
    _images(exp_dir, folder="more")
    other = tmp_path / "elsewhere"
    other.mkdir()
    monkeypatch.chdir(other)  # a relative path must not depend on the working directory
    data = {"images": "imgs", "nut": {"host": "h", "port": 1}, "seeds": [1], "output": "out/run",
            "methods": {"mio": {"algorithm": "mio"}}}
    exp = load_experiment(_write(exp_dir, data))
    assert exp.targets[0].images == str(exp_dir / "imgs")
    assert exp.output == str(exp_dir / "out" / "run")
    data = {"targets": [{"name": "a", "images": "imgs", "nut": {"host": "h", "port": 1}},
                        {"name": "b", "images": str(exp_dir / "more"), "nut": {"host": "h", "port": 2}}],
            "seeds": [1], "output": str(tmp_path / "abs_out"), "methods": {"mio": {"algorithm": "mio"}}}
    exp = load_experiment(_write(exp_dir, data, name="multi.yaml"))
    assert [t.images for t in exp.targets] == [str(exp_dir / "imgs"), str(exp_dir / "more")]
    assert exp.output == str(tmp_path / "abs_out")
