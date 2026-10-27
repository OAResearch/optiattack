"""Summary tables and paired statistics for a results.csv written by the experiment runner."""
import csv
import math
import os
from itertools import combinations
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from core.experiment import stats

DEFAULT_K_VALUES = [10, 20, 50, 100]
METRICS = ["l0_pixels", "l2", "linf", "ssim", "psnr"]
NO_TIME_BUDGET = "n/a (time budget)"
NO_BUDGET = "n/a (unknown budget)"
TIME_CRITERION = "time"
Row = Dict[str, str]
Key = Tuple[str, str, str]

PAIR_COLUMNS = [("pair", "pair"), ("n_pairs", "pairs"), ("q_p", "queries Wilcoxon p"),
                ("q_p_holm", "queries Holm p"), ("q_better", "P(first fewer queries) = 1-A12"),
                ("l0_n", "L0 pairs"), ("l0_p", "L0 Wilcoxon p"), ("l0_p_holm", "L0 Holm p"),
                ("l0_better", "P(first lower L0) = 1-A12"), ("mcnemar_p", "McNemar p (success)"),
                ("mcnemar_p_holm", "McNemar Holm p")]


def num(value: Optional[str]) -> Optional[float]:
    """Parse a CSV cell to a float; empty or invalid cells become None."""
    if value is None or value.strip() == "":
        return None
    try:
        number = float(value)
    except ValueError:
        return None
    return None if math.isnan(number) else number


def _flipped(row: Row) -> bool:
    """True if the run succeeded."""
    return row.get("flipped", "").strip().lower() == "true"


def read_results(path: str) -> List[Row]:
    """Read results.csv into a list of rows."""
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _queries_of(row: Row) -> Optional[float]:
    """Queries until the first success; falls back to eval_count if the column is empty."""
    value = num(row.get("first_success_eval"))
    return value if value is not None else num(row.get("eval_count"))


def _median(values: Sequence[float]) -> Optional[float]:
    """Median or None for no values."""
    return float(np.median(values)) if values else None


def _collect(rows: Sequence[Row], column: str) -> List[float]:
    """Numeric values of a column in the given rows (empty cells skipped)."""
    return [v for v in (num(r.get(column)) for r in rows) if v is not None]


def _within_k(row: Row, k: int) -> bool:
    """True if the run has an l0_pixels value of at most k."""
    l0 = num(row.get("l0_pixels"))
    return l0 is not None and l0 <= k


def method_row(rows: Sequence[Row], k_values: Sequence[int]) -> Dict[str, Any]:
    """Summary columns for the runs of one (target, method)."""
    n = len(rows)
    ok = [r for r in rows if _flipped(r)]
    queries = [q for q in (_queries_of(r) for r in ok) if q is not None]
    out: Dict[str, Any] = {"n_runs": n, "success_rate": len(ok) / n if n else None,
                           "queries_median": _median(queries),
                           "queries_q1": float(np.percentile(queries, 25)) if queries else None,
                           "queries_q3": float(np.percentile(queries, 75)) if queries else None}
    for metric in METRICS:
        out[metric + "_median"] = _median(_collect(ok, metric))
    out["pruning_eval_count_median"] = _median(_collect(rows, "pruning_eval_count"))
    for k in k_values:
        out[f"success@{k}"] = sum(1 for r in ok if _within_k(r, k)) / n if n else None
    return out


def _index(rows: Sequence[Row], method: str) -> Dict[Key, Row]:
    """Rows of one method keyed by (target, image, seed)."""
    return {(r["target"], r["image"], r["seed"]): r for r in rows if r["method"] == method}


def _matched(rows: Sequence[Row], a: str, b: str) -> List[Tuple[Row, Row]]:
    """Run pairs of two methods matched on (target, image, seed), in a stable order."""
    ia, ib = _index(rows, a), _index(rows, b)
    return [(ia[k], ib[k]) for k in sorted(set(ia) & set(ib))]


def is_time_budget(row: Row) -> bool:
    """True if the run stopped on a time budget (config_stopping_criterion 'time')."""
    return (row.get("config_stopping_criterion") or "").strip().lower() == TIME_CRITERION


def row_budget(row: Row, default: Optional[int] = None) -> Optional[int]:
    """Query budget of one run from its config_max_evaluations; default if absent, None for a time budget."""
    if is_time_budget(row):
        return None
    max_evals = num(row.get("config_max_evaluations"))
    return int(max_evals) if max_evals is not None else default


def _needs_penalty(row: Row) -> bool:
    """True if the run's query cost is its budget penalty (failed, or no query count)."""
    return not _flipped(row) or _queries_of(row) is None


def paired_queries(rows: Sequence[Row], a: str, b: str,
                   default_budget: Optional[int] = None) -> List[Tuple[float, float]]:
    """Paired query counts of two methods; a failed run counts as its own budget + 1."""
    def cost(row: Row) -> float:
        queries = _queries_of(row) if _flipped(row) else None
        if queries is not None:
            return queries
        budget = row_budget(row, default_budget)
        if budget is None:
            raise ValueError(f"run {row.get('method')}/{row.get('image')}/seed {row.get('seed')} has no query budget")
        return float(budget + 1)

    return [(cost(ra), cost(rb)) for ra, rb in _matched(rows, a, b)]


def _query_test_status(matched: Sequence[Tuple[Row, Row]], default_budget: Optional[int]) -> Optional[str]:
    """Reason the query test of a pair cannot run (time budget, unknown budget), or None if it can."""
    runs = [r for pair in matched for r in pair]
    if any(is_time_budget(r) for r in runs):
        return NO_TIME_BUDGET
    if any(_needs_penalty(r) and row_budget(r, default_budget) is None for r in runs):
        return NO_BUDGET
    return None


def _budget_note(matched: Sequence[Tuple[Row, Row]], a: str, b: str, default_budget: Optional[int]) -> Optional[str]:
    """Note for a pair whose two methods ran with different query budgets, else None."""
    def budgets(runs: Sequence[Row]) -> str:
        values = sorted({row_budget(r, default_budget) for r in runs if not is_time_budget(r)},
                        key=lambda v: -1 if v is None else v)
        return ", ".join("unknown" if v is None else str(v) for v in values) or "time budget"

    first, second = budgets([ra for ra, _ in matched]), budgets([rb for _, rb in matched])
    if first == second:
        return None
    return (f"{a} vs {b}: different query budgets ({a}: {first}; {b}: {second}). Each failed run is penalized "
            f"with its own budget + 1, so the query comparison is not like-for-like.")


def paired_l0(rows: Sequence[Row], a: str, b: str) -> List[Tuple[float, float]]:
    """Paired L0 values, only for pairs where both runs succeeded."""
    out = []
    for ra, rb in _matched(rows, a, b):
        if _flipped(ra) and _flipped(rb):
            la, lb = num(ra.get("l0_pixels")), num(rb.get("l0_pixels"))
            if la is not None and lb is not None:
                out.append((la, lb))
    return out


def _test(pairs: Sequence[Tuple[float, float]]) -> Dict[str, Any]:
    """Wilcoxon p and 'first method is better' probability (1 - A12, lower is better) of paired values."""
    x, y = [p[0] for p in pairs], [p[1] for p in pairs]
    return {"n": len(pairs), "p": stats.wilcoxon_paired(x, y) if pairs else None,
            "better": 1.0 - stats.a12(x, y) if pairs else None}


def pairwise(rows: Sequence[Row], methods: Sequence[str],
             default_budget: Optional[int] = None) -> List[Dict[str, Any]]:
    """Pairwise comparisons of one group (Holm per test family); budgets come from each row's config columns."""
    # default_budget only fills rows without config_max_evaluations.
    present = [m for m in methods if any(r["method"] == m for r in rows)]
    result: List[Dict[str, Any]] = []
    for a, b in combinations(present, 2):
        matched = _matched(rows, a, b)
        entry: Dict[str, Any] = {"pair": f"{a} vs {b}", "n_pairs": len(matched),
                                 "budget_note": _budget_note(matched, a, b, default_budget)}
        status = _query_test_status(matched, default_budget)
        if status is not None:
            entry.update({"q_p": status, "q_better": status})
        else:
            q = _test(paired_queries(rows, a, b, default_budget))
            entry.update({"q_p": q["p"], "q_better": q["better"]})
        lz = _test(paired_l0(rows, a, b))
        entry.update({"l0_n": lz["n"], "l0_p": lz["p"], "l0_better": lz["better"]})
        entry["mcnemar_p"] = (stats.mcnemar_exact([_flipped(ra) for ra, _ in matched],
                                                  [_flipped(rb) for _, rb in matched]) if matched else None)
        result.append(entry)
    for col in ("q_p", "l0_p", "mcnemar_p"):
        raw = {e["pair"]: e[col] for e in result if isinstance(e[col], float)}
        adjusted = stats.holm(raw)
        for e in result:
            e[col + "_holm"] = e[col] if isinstance(e[col], str) else adjusted.get(e["pair"])
    return result


def _fmt(value: Any) -> str:
    """Format a table cell for Markdown."""
    if value is None:
        return ""
    if isinstance(value, float):
        return "inf" if math.isinf(value) else f"{value:.4g}"
    return str(value)


def _fmt_csv(value: Any) -> str:
    """Format a CSV cell: blank for None, full precision for floats."""
    return "" if value is None else repr(value) if isinstance(value, float) else str(value)


def _md_table(header: Sequence[str], body: Sequence[Sequence[Any]]) -> List[str]:
    """Render a Markdown table."""
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    lines += ["| " + " | ".join(_fmt(c) for c in r) + " |" for r in body]
    return lines + [""]


def _group_markdown(name: str, methods: Sequence[str], rows: Sequence[Row],
                    default_budget: Optional[int]) -> List[str]:
    """Markdown section for one threat-model group."""
    lines = [f"### {name}", ""]
    present = {r["method"] for r in rows}
    missing = [m for m in methods if m not in present]
    lines += [f"- {m}: no runs" for m in missing]
    if missing:
        lines.append("")
    entries = pairwise(rows, methods, default_budget)
    if entries:
        lines += _md_table([h for _, h in PAIR_COLUMNS], [[e.get(k) for k, _ in PAIR_COLUMNS] for e in entries])
        notes = [f"- **Note:** {e['budget_note']}" for e in entries if e.get("budget_note")]
        lines += notes + ([""] if notes else [])
    else:
        lines += ["Fewer than two methods with runs; nothing to compare.", ""]
    return lines


def _write_csv(path: str, columns: Sequence[str], entries: Sequence[Dict[str, Any]]) -> None:
    """Write dict entries as a CSV with the given columns."""
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(columns)
        for entry in entries:
            writer.writerow([_fmt_csv(entry.get(c)) for c in columns])


def _penalty_text(rows: Sequence[Row], default_budget: Optional[int]) -> str:
    """Describe the query penalty of the results (per-run budgets, time budgets)."""
    budgets = sorted({b for b in (row_budget(r, default_budget) for r in rows) if b is not None})
    text = ("a failed run counts as its own budget + 1 queries ("
            + (", ".join(str(b + 1) for b in budgets) if budgets else "no query budget recorded") + ")")
    n_time = sum(1 for r in rows if is_time_budget(r))
    if n_time:
        text += f"; {n_time} runs used a time budget, so query tests involving them are {NO_TIME_BUDGET}"
    return text


def summarize(results_csv: str, analysis: Dict[str, Any], default_budget: Optional[int], out_dir: str) -> str:
    """Write summary.md, summary.csv and pairwise.csv into out_dir; query budgets come from each row's config."""
    # Budgets: config_max_evaluations / config_stopping_criterion; default_budget only fills rows without them.
    rows = read_results(results_csv)
    k_values = [int(k) for k in (analysis.get("k_values") or DEFAULT_K_VALUES)]
    groups = {str(g): [str(m) for m in ms] for g, ms in (analysis.get("threat_models") or {}).items()}
    os.makedirs(out_dir, exist_ok=True)

    table = []
    for target, method in sorted({(r["target"], r["method"]) for r in rows}):
        sub = [r for r in rows if r["target"] == target and r["method"] == method]
        table.append({"target": target, "method": method, **method_row(sub, k_values)})
    columns = (["target", "method", "n_runs", "success_rate", "queries_median", "queries_q1", "queries_q3"]
               + [m + "_median" for m in METRICS] + ["pruning_eval_count_median"] + [f"success@{k}" for k in k_values])
    _write_csv(os.path.join(out_dir, "summary.csv"), columns, table)

    lines = ["# Experiment summary", "", f"Runs: {len(rows)}. Query penalty: {_penalty_text(rows, default_budget)}.",
             "", "## Per method", ""]
    lines += _md_table(columns, [[entry.get(c) for c in columns] for entry in table])
    lines += ["Queries, L0, L2, Linf, SSIM and PSNR are medians over successful runs; success@k is the fraction of "
              "all runs that succeeded with l0_pixels <= k.", "", "## Paired comparisons", "",
              "Holm adjustment is applied per threat-model group, separately for each test family "
              "(queries, L0, success).", ""]
    if not groups:
        lines += ["No threat models configured (analysis.threat_models); no comparisons.", ""]
    pair_rows: List[Dict[str, Any]] = []
    for name, methods in groups.items():
        lines += _group_markdown(name, methods, rows, default_budget)
        pair_rows += [{"group": name, **e} for e in pairwise(rows, methods, default_budget)]
    if pair_rows:
        _write_csv(os.path.join(out_dir, "pairwise.csv"),
                   ["group"] + [k for k, _ in PAIR_COLUMNS] + ["budget_note"], pair_rows)
    md_path = os.path.join(out_dir, "summary.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    return md_path
