"""Tests for the experiment summary."""
import csv
import os

import pytest

pytest.importorskip("scipy")

from core.experiment import summary  # noqa: E402

COLUMNS = ["target", "method", "image", "seed", "flipped", "first_success_eval", "eval_count", "l0_pixels", "l2",
           "linf", "ssim", "psnr", "pruning_eval_count"]


def write_results(path, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(COLUMNS)
        for r in rows:
            w.writerow(r)
    return str(path)


def run(tmp_path, rows, analysis=None, budget=100):
    results = write_results(tmp_path / "results.csv", rows)
    out = tmp_path / "out"
    md = summary.summarize(results, analysis or {"k_values": [5, 10]}, budget, str(out))
    with open(out / "summary.csv", newline="", encoding="utf-8") as f:
        table = list(csv.DictReader(f))
    with open(md, encoding="utf-8") as f:
        return table, f.read()


def ok(method, image, seed, first, l0, target="t"):
    return [target, method, image, seed, "True", first, first + 5, l0, "1.5", "0.2", "0.9", "inf", 7]


def fail(method, image, seed, target="t"):
    return [target, method, image, seed, "False", "", 100, "", "", "", "", "", ""]


def test_success_at_k(tmp_path):
    rows = [ok("a", "i1", 1, 10, 4), ok("a", "i2", 1, 20, 8), ok("a", "i3", 1, 30, 12), fail("a", "i4", 1)]
    table, _ = run(tmp_path, rows)
    row = table[0]
    assert row["n_runs"] == "4"
    assert float(row["success_rate"]) == 0.75
    assert float(row["success@5"]) == 0.25
    assert float(row["success@10"]) == 0.5
    assert float(row["queries_median"]) == 20
    assert float(row["queries_q1"]) == 15
    assert float(row["queries_q3"]) == 25
    assert float(row["l0_pixels_median"]) == 8


def test_summary_with_method_that_never_succeeds(tmp_path):
    rows = [fail("a", "i1", 1), fail("a", "i2", 1), ok("b", "i1", 1, 10, 3)]
    table, text = run(tmp_path, rows)
    a = [r for r in table if r["method"] == "a"][0]
    assert float(a["success_rate"]) == 0
    assert a["queries_median"] == "" and a["l2_median"] == "" and a["psnr_median"] == ""
    assert "a" in text


def test_failed_runs_penalized_with_budget_plus_one(tmp_path):
    # A succeeds in 10+ queries on 4 images; B fails on all, so each of B's runs is penalized with budget + 1 = 101.
    rows = []
    for i in range(1, 5):
        rows.append(ok("a", f"i{i}", 1, 10 + i, 3))
        rows.append(fail("b", f"i{i}", 1))
    pairs = summary.paired_queries(summary.read_results(write_results(tmp_path / "r.csv", rows)), "a", "b", 100)
    assert [p[1] for p in pairs] == [101.0] * 4
    assert [p[0] for p in pairs] == [11.0, 12.0, 13.0, 14.0]


def test_threat_model_tables_are_separate(tmp_path):
    rows = []
    for m, base in (("a", 10), ("b", 20), ("c", 30), ("d", 40)):
        for i in range(1, 7):
            rows.append(ok(m, f"i{i}", 1, base + i, 3 + i))
    analysis = {"k_values": [5], "threat_models": {"l0": ["a", "b"], "linf": ["c", "d", "ghost"]}}
    table, text = run(tmp_path, rows, analysis)
    assert "### l0" in text and "### linf" in text
    l0_part = text.split("### l0")[1].split("### linf")[0]
    linf_part = text.split("### linf")[1]
    assert "a vs b" in l0_part and "c vs d" not in l0_part
    assert "c vs d" in linf_part and "a vs b" not in linf_part
    assert "ghost" in linf_part and "no runs" in linf_part
    assert os.path.isfile(tmp_path / "out" / "summary.md")


def test_rows_without_budget_columns(tmp_path):
    # every run succeeded: no penalty is needed, so the query test runs without any budget
    rows = [ok("a", f"i{i}", 1, 10 + i, 3) for i in range(1, 5)] + [ok("b", f"i{i}", 1, 30 + i, 4) for i in range(1, 5)]
    entries = summary.pairwise(summary.read_results(write_results(tmp_path / "r.csv", rows)), ["a", "b"])
    assert isinstance(entries[0]["q_p"], float)
    # a failed run without a budget cannot be penalized: the query test is skipped, not guessed
    rows[-1] = fail("b", "i4", 1)
    _, text = run(tmp_path, rows, {"threat_models": {"g": ["a", "b"]}}, budget=None)
    assert "n/a (unknown budget)" in text and "n/a (time budget)" not in text


def test_cli_summarize_without_yaml_uses_defaults(tmp_path):
    from core.experiment.__main__ import main
    results = write_results(tmp_path / "results.csv", [ok("a", "i1", 1, 10, 4), fail("a", "i2", 1)])
    assert main(["summarize", results]) == 0
    with open(tmp_path / "summary.csv", newline="", encoding="utf-8") as f:
        row = next(csv.DictReader(f))
    assert "success@100" in row and row["n_runs"] == "2"


def test_cli_summarize_with_yaml_reads_analysis_and_budget(tmp_path):
    pytest.importorskip("yaml")
    from core.experiment.__main__ import main
    (tmp_path / "imgs").mkdir()
    (tmp_path / "imgs" / "i1.png").write_bytes(b"x")
    yaml_path = tmp_path / "e.yaml"
    yaml_path.write_text(
        f"images: {(tmp_path / 'imgs').as_posix()}\nnut: {{host: h, port: 1}}\nbudget: {{max_evaluations: 50}}\n"
        f"seeds: [1]\noutput: {(tmp_path / 'o').as_posix()}\nmethods: {{a: {{algorithm: mio}}}}\n"
        "analysis: {k_values: [3]}\n", encoding="utf-8")
    results = write_results(tmp_path / "results.csv", [ok("a", "i1", 1, 10, 4)])
    assert main(["summarize", results, "--yaml", str(yaml_path)]) == 0
    with open(tmp_path / "summary.csv", newline="", encoding="utf-8") as f:
        row = next(csv.DictReader(f))
    assert "success@3" in row and "success@100" not in row


def test_pairwise_statistics_with_three_methods(tmp_path):
    from scipy.stats import binomtest
    rows = []
    for i in range(1, 7):
        rows.append(ok("a", f"i{i}", 1, 10 + i, 3))
    for i in range(1, 5):
        rows.append(ok("b", f"i{i}", 1, 20 + i, 5))
    rows += [fail("b", "i5", 1), fail("b", "i6", 1)]
    rows.append(ok("c", "i1", 1, 5, 2))
    entries = summary.pairwise(summary.read_results(write_results(tmp_path / "r.csv", rows)), ["a", "b", "c"], 100)
    by_pair = {e["pair"]: e for e in entries}
    ab = by_pair["a vs b"]
    assert ab["n_pairs"] == 6
    assert ab["q_better"] > 0.5
    assert ab["l0_n"] == 4  # i5 and i6 are excluded because b failed there
    assert ab["mcnemar_p"] == pytest.approx(binomtest(2, 2).pvalue)
    assert ab["mcnemar_p"] == 0.5
    for e in entries:
        for col in ("q_p", "l0_p", "mcnemar_p"):
            if e[col] is None:
                assert e[col + "_holm"] is None
            else:
                assert e[col] <= e[col + "_holm"] <= 1.0
    ac = by_pair["a vs c"]
    assert ac["n_pairs"] == 1 and ac["q_p"] is None and ac["l0_p"] is None and ac["q_p_holm"] is None
    assert ac["mcnemar_p"] == 1.0
    assert "" == summary._fmt(ac["q_p"])


PLAIN_METHODS = "methods: {a: {algorithm: mio}, b: {algorithm: mio}}"
CONFIG_COLUMNS = COLUMNS + ["config_max_evaluations", "config_stopping_criterion"]


def write_config_results(path, rows):
    """rows: (row, max_evaluations, stopping_criterion) triples."""
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(CONFIG_COLUMNS)
        for row, max_evals, criterion in rows:
            w.writerow(row + [max_evals, criterion])
    return str(path)


def _ab_rows(budget_a, budget_b, criterion_a="individual_evaluations", criterion_b="individual_evaluations"):
    rows = []
    for i in range(1, 5):
        rows.append((ok("a", f"i{i}", 1, 10 + i, 3), budget_a, criterion_a))
        rows.append((fail("b", f"i{i}", 1), budget_b, criterion_b))
    return rows


def _experiment_yaml(tmp_path, params):
    (tmp_path / "imgs").mkdir(exist_ok=True)
    (tmp_path / "imgs" / "i1.png").write_bytes(b"x")
    yaml_path = tmp_path / "e.yaml"
    yaml_path.write_text(
        f"images: {(tmp_path / 'imgs').as_posix()}\nnut: {{host: h, port: 1}}\nseeds: [1]\n"
        f"output: {(tmp_path / 'o').as_posix()}\n{params}\n"
        "analysis: {threat_models: {g: [a, b]}}\n", encoding="utf-8")
    return str(yaml_path)


def _cli_summary(tmp_path, rows, yaml_params=None):
    from core.experiment.__main__ import main
    results = write_config_results(tmp_path / "results.csv", rows)
    argv = ["summarize", results]
    if yaml_params is not None:
        pytest.importorskip("yaml")
        argv += ["--yaml", _experiment_yaml(tmp_path, yaml_params)]
    assert main(argv) == 0
    pair = None
    if os.path.isfile(tmp_path / "pairwise.csv"):
        with open(tmp_path / "pairwise.csv", newline="", encoding="utf-8") as f:
            pair = next(csv.DictReader(f))
    with open(tmp_path / "summary.md", encoding="utf-8") as f:
        return pair, f.read()


def test_penalty_uses_each_rows_budget(tmp_path):
    rows = summary.read_results(write_config_results(tmp_path / "r.csv", _ab_rows(50, 200)))
    pairs = summary.paired_queries(rows, "a", "b")
    assert [p[1] for p in pairs] == [201.0] * 4
    assert [p[0] for p in pairs] == [11.0, 12.0, 13.0, 14.0]
    pairs = summary.paired_queries(rows, "b", "a")
    assert [p[0] for p in pairs] == [201.0] * 4


def test_cli_budget_per_method(tmp_path):
    params = "methods: {a: {algorithm: mio, max_evaluations: 300}, b: {algorithm: mio, max_evaluations: 300}}"
    pair, text = _cli_summary(tmp_path, _ab_rows(300, 300), params)
    assert "n/a" not in pair["q_p"] and float(pair["q_p"]) > 0
    assert "n/a (time budget)" not in text and "301" in text


def test_cli_budget_in_common(tmp_path):
    params = "common: {max_evaluations: 300}\nmethods: {a: {algorithm: mio}, b: {algorithm: mio}}"
    pair, text = _cli_summary(tmp_path, _ab_rows(300, 300), params)
    assert "n/a" not in pair["q_p"] and float(pair["q_p"]) > 0
    assert "n/a (time budget)" not in text and "301" in text


def test_cli_missing_budget_falls_back_to_rows_config(tmp_path):
    pair, text = _cli_summary(tmp_path, _ab_rows(1000, 1000), PLAIN_METHODS)
    assert "n/a" not in pair["q_p"] and float(pair["q_p"]) > 0
    assert "1001" in text and "n/a (time budget)" not in text


def test_cli_without_yaml_uses_rows_budget_not_max_eval_count(tmp_path):
    pair, text = _cli_summary(tmp_path, _ab_rows(1000, 1000))
    # the failed runs have eval_count 100; the old guess would have penalized them with 101
    assert "1001" in text and "101 queries" not in text


def test_time_budget_rows_skip_query_test(tmp_path):
    pair, text = _cli_summary(tmp_path, _ab_rows(1000, 1000, "time", "time"), PLAIN_METHODS)
    assert pair["q_p"] == "n/a (time budget)" and pair["q_p_holm"] == "n/a (time budget)"
    assert "n/a (time budget)" in text


def test_evaluation_budget_with_yaml_time_criterion_absent_is_not_time_budget(tmp_path):
    # budget: {} in the YAML used to be read as a time budget
    pair, _ = _cli_summary(tmp_path, _ab_rows(500, 500), "budget: {}\nmethods: {a: {algorithm: mio}, b: {}}")
    assert pair["q_p"] != "n/a (time budget)"


def test_mixed_budget_pair_gets_a_note(tmp_path):
    pair, text = _cli_summary(tmp_path, _ab_rows(50, 200), PLAIN_METHODS)
    assert float(pair["q_p"]) > 0
    note = [line for line in text.splitlines() if "a vs b" in line and "budget" in line.lower()]
    assert note and "50" in note[0] and "200" in note[0]
