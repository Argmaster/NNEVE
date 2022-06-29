"""CLI wrapper package."""


from pathlib import Path
from typing import Any, List

import click

from nneve import __version__
from nneve.common.logconfig import configure_logger

DIR = Path(__file__).parent


def cli(args: List[str]) -> Any:
    """NNEVE CLI interface API endpoint."""
    return nneve(args)


@click.group()
@click.version_option(
    __version__,
    "--version",
    "-V",
    package_name="NNEVE",
)
@click.option(
    "-d",
    "--debug",
    default=False,
    is_flag=True,
    help="Enable debug mode, implies verbose logging.",
)
@click.option(
    "-v",
    "--verbose",
    default=False,
    is_flag=True,
    help="Enable verbose logging, do not implies debug mode.",
)
def nneve(debug: bool, verbose: bool) -> None:
    """nneve entry point description."""
    configure_logger(debug, verbose)
