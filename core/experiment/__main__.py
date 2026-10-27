"""Command line entry point: python -m core.experiment run|summarize."""
import argparse
import logging
import os
import sys
from typing import Any, Dict, List, Optional

from core.experiment.config import ExperimentConfigError, load_experiment


def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser."""
    parser = argparse.ArgumentParser(prog="python -m core.experiment", description="OptiAttack batch experiments")
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run", help="run every target x method x image x seed of an experiment YAML")
    run.add_argument("yaml", help="experiment YAML file")
    summarize = sub.add_parser("summarize", help="summarize a results.csv and compare methods")
    summarize.add_argument("results", help="results.csv written by 'run'")
    summarize.add_argument("--yaml", default=None, help="experiment YAML (for k values and threat models)")
    return parser


def _summarize_file(results: str, yaml_path: Optional[str]) -> str:
    """Summarize results.csv; analysis comes from the YAML when given, query budgets from each row's config."""
    analysis: Dict[str, Any] = load_experiment(yaml_path).analysis if yaml_path else {}
    return _summarize_experiment(results, analysis, os.path.dirname(os.path.abspath(results)))


def _summarize_experiment(results: str, analysis: Dict[str, Any], out_dir: str) -> str:
    """Summarize results.csv; each run's query budget comes from its config_max_evaluations column."""
    from core.experiment import summary
    return summary.summarize(results, analysis, None, out_dir)


def main(argv: Optional[List[str]] = None) -> int:
    """Run the command line; returns the process exit code."""
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    if args.command == "summarize":
        try:
            print(f"Summary: {_summarize_file(args.results, args.yaml)}")
        except (ExperimentConfigError, ImportError, OSError, KeyError) as e:
            print(f"Error: {e}", file=sys.stderr)
            return 1
        return 0
    try:
        from core.experiment.runner import run_experiment
        experiment = load_experiment(args.yaml)
        results = run_experiment(experiment)
        summary_path = _summarize_experiment(results, experiment.analysis, experiment.output)
    except (ExperimentConfigError, ImportError, OSError) as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    print(f"Results: {results}")
    print(f"Summary: {summary_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
