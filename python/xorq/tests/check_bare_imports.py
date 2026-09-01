"""Import every module in bare_install_modules.txt; exit non-zero on the first failure.

Run inside an environment holding only [project].dependencies, so it must not
import anything beyond the standard library and xorq itself.
"""

import importlib
import pathlib
import sys


def main():
    listing = pathlib.Path(__file__).with_name("bare_install_modules.txt")
    lines = (line.strip() for line in listing.read_text().splitlines())
    names = [line for line in lines if line and not line.startswith("#")]
    if not names:
        sys.exit(f"no modules listed in {listing}")
    for name in names:
        importlib.import_module(name)
        print(f"ok {name}")
    print(f"imported {len(names)} modules with declared dependencies only")


if __name__ == "__main__":
    main()
