import pytest

from nneve.cli import cli


def test_cli_version() -> None:
    with pytest.raises(SystemExit, match="0"):
        cli(["--version"])
