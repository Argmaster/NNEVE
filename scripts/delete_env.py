#!/usr/bin/python3
import argparse
import logging
import shutil
import sys
from pathlib import Path
from typing import Iterable, List

TOX_LOCATIONS = (
    Path(".") / ".tox",
    Path(__file__).parent / ".." / ".tox",
)


def purge_tox_env_cli(args: List[str]):
    parser = argparse.ArgumentParser("delete_env.py")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--all", "-a", action="store_true", default=False)
    group.add_argument("--list", "-l", nargs="+", default=[])
    parser.add_argument("--yes", "-y", action="store_false", default=True)
    namespace = parser.parse_args(args)
    return purge_tox_env(namespace.list, namespace.all, namespace.yes)


def purge_tox_env(
    envs_list: List[str],
    delete_all: bool = False,
    ask_for_approval: bool = True,
):
    if envs_list is None and delete_all is False:
        print(
            "Neither environment name nor --all flag was given: no purge done."
        )
        return 0

    if delete_all is False:
        for env in envs_list:
            return _purge_env(env, ask_for_approval)
    else:
        return _purge_all(ask_for_approval)


def _purge_env(name: str, ask_for_approval: bool):
    for possible_path in _get_tox_locations():
        env_path: Path = possible_path / name
        if not env_path.exists():
            continue
        if ask_for_approval and (
            input(f"Delete {env_path}? (y/n) ").lower() != "y"
        ):
            break
        _delete(env_path)
        logging.warning(f"Removed {env_path}")
        break
    else:
        logging.warning("Environment not found.")
        return 3


def _get_tox_locations() -> Iterable[Path]:
    return filter(lambda loc: loc.is_dir(), TOX_LOCATIONS)


def _delete(path: Path) -> None:
    shutil.rmtree(str(path))


def _purge_all(ask_for_approval: bool):
    for tox_path in _get_tox_locations():
        if ask_for_approval and (
            input(f"Delete {tox_path}? (y/n)").lower() != "y"
        ):
            break
        _delete(tox_path)
        logging.warning(f"Removed {tox_path}")
        break
    else:
        logging.warning("Tox directory not found.")
        return 4
    return 0


if __name__ == "__main__":
    raise SystemExit(purge_tox_env_cli(sys.argv[1:]))
