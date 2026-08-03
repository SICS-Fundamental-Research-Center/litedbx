"""Command-line entrance for LiteDBX experiment configs."""

import argparse
import asyncio
import logging
import sys
from collections.abc import Sequence
from pathlib import Path
from time import time

from exp.experiment_runner import (
    export_results,
    run_config,
)

ROOT_DIR = Path(__file__).parent
EXP_DIR = ROOT_DIR / "exp"
DEFAULT_CONFIG = "default.yaml"
LOG_FORMAT = (
    "%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s"
)


def configure_logging() -> logging.Logger:
    """Configure process logging and return the module logger."""
    logging.basicConfig(
        level=logging.INFO,
        format=LOG_FORMAT,
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    return logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser for config-based execution."""
    parser = argparse.ArgumentParser(description="LiteDBX entrance")
    parser.add_argument(
        "--debug",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable debug execution for tasks that do not set debug.",
    )
    parser.add_argument(
        "--config",
        action="append",
        help="Experiment config path or file under exp/.",
    )
    parser.add_argument(
        "--cold",
        action="store_true",
        help="Disable cache reads and writes for this run.",
    )
    parser.add_argument(
        "--ls-configs",
        action="store_true",
        help="List experiment configs under exp/.",
    )
    return parser


def has_collectable_rows(results: Sequence[object]) -> bool:
    """Return whether any task result has rows to export."""
    return any(getattr(result, "rows", None) for result in results)


async def run_configs(
    config_names: list[str],
    debug: bool,
    logger: logging.Logger,
    cold: bool = False,
) -> None:
    """Run experiment configs and export results when rows are collected."""
    for config_name in config_names:
        config_path = Path(config_name)
        if not config_path.exists():
            config_path = EXP_DIR / "configs" / config_path
        if not config_path.exists():
            raise FileNotFoundError(f"Config not found: {config_name}")

        results = await run_config(config_path, debug=debug, cold=cold)

        if has_collectable_rows(results):
            out_path = export_results(config_path, results)
            logger.info("Exported experiment results to %s", out_path)
        else:
            logger.info("No experiment rows collected for %s", config_path)


def main() -> None:
    """Parse arguments and execute the requested config path."""
    logger = configure_logging()
    args = build_parser().parse_args()

    if args.ls_configs:
        config_paths = sorted(p for p in (EXP_DIR / "configs").glob("*.yaml"))

        for path in config_paths:
            print(path.name)
        return

    config_names = args.config or [DEFAULT_CONFIG]

    start = time()
    asyncio.run(run_configs(config_names, args.debug, logger, cold=args.cold))
    end = time()

    logger.info("Total execution time: %s seconds", end - start)


if __name__ == "__main__":
    main()
