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
# Experiment config groups, listed by --ls-configs. --config takes a path
# relative to exp/ in the "<group>/<file>" form these listings print (e.g.
# configs/default.yaml, static_bound_calibration/sbc_image.yaml); literal and
# absolute paths are also accepted. Bare filenames are intentionally NOT
# searched across groups, so identically named configs cannot collide.
CONFIG_DIRS = (
    EXP_DIR / "configs",
    EXP_DIR / "static_bound_calibration",
)
DEFAULT_CONFIG = "configs/default.yaml"
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
        help=(
            "Experiment config: a path relative to exp/ in <group>/<file> "
            "form (e.g. configs/default.yaml, "
            "static_bound_calibration/sbc_image.yaml); literal/absolute "
            "paths also accepted. See --ls-configs."
        ),
    )
    parser.add_argument(
        "--cold",
        action="store_true",
        help="Disable cache reads and writes for this run.",
    )
    parser.add_argument(
        "--certificate",
        action="store_true",
        help="Compute error certificates after running the workload.",
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


def resolve_config_path(config_name: str) -> Path:
    """Resolve an experiment config path.

    Accepts, in order:
      1. A literal path that exists relative to the cwd (covers absolute paths
         and forms like exp/configs/default.yaml).
      2. A path relative to exp/ in "<group>/<file>" form -- the canonical
         form ``--ls-configs`` prints (e.g. configs/default.yaml,
         static_bound_calibration/sbc_image.yaml).

    Bare filenames are NOT searched across config groups: identically named
    configs in different groups must not collide, so the caller gives the full
    path from the experiment group to the file.
    """
    config_path = Path(config_name)
    if config_path.exists():
        return config_path
    exp_relative = EXP_DIR / config_path
    if exp_relative.exists():
        return exp_relative
    raise FileNotFoundError(
        f"Config not found: {config_name!r}. Give the path relative to exp/ "
        f"in <group>/<file> form (e.g. configs/default.yaml or "
        f"static_bound_calibration/sbc_image.yaml); see --ls-configs."
    )


async def run_configs(
    config_names: list[str],
    debug: bool,
    logger: logging.Logger,
    cold: bool = False,
    certificate: bool = False,
) -> None:
    """Run experiment configs and export results when rows are collected."""
    for config_name in config_names:
        config_path = resolve_config_path(config_name)

        results = await run_config(
            config_path, debug=debug, cold=cold, certificate=certificate)

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
        seen = set()
        for candidate_dir in CONFIG_DIRS:
            for path in sorted(candidate_dir.glob("*.yaml")):
                if path.name in seen:
                    continue
                seen.add(path.name)
                print(path.relative_to(EXP_DIR))
        return

    config_names = args.config or [DEFAULT_CONFIG]

    start = time()
    asyncio.run(run_configs(
        config_names, 
        args.debug, 
        logger, 
        cold=args.cold, 
        certificate=args.certificate
    ))
    end = time()

    logger.info("Total execution time: %s seconds", end - start)


if __name__ == "__main__":
    main()
