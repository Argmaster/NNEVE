"""CLI entry point."""
import sys

from .cli import cli

cli(sys.argv[1:])
