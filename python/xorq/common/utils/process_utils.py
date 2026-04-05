import functools
import os
import re
import subprocess


@functools.cache
def in_nix_shell():
    return bool(os.environ.get("IN_NIX_SHELL"))


def assert_not_in_nix_shell():
    if in_nix_shell():
        raise ValueError("in nix shell")


def subprocess_run(args, text=False, **kwargs):
    result = subprocess.run(
        args,
        capture_output=True,
        encoding="utf-8" if text else None,
        **kwargs,
    )
    return (result.returncode, result.stdout, result.stderr)


# https://stackoverflow.com/a/14693789
ansi_escape = re.compile(r"\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])")
remove_ansi_escape = functools.partial(ansi_escape.sub, "")
