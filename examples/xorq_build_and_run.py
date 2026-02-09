#!/usr/bin/env python
"""xorq_build_and_run.py — build a xorq expression and immediately run it

Usage:
    python xorq_build_and_run.py <script.py> [-e <expr_name>] [-o <output_path>] [-f <format>]

Examples:
    python xorq_build_and_run.py simple_example.py
    python xorq_build_and_run.py simple_example.py -e expr
    python xorq_build_and_run.py iris_example.py -e expr -o results.parquet
    python xorq_build_and_run.py simple_example.py -f json -o -
"""

import argparse
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(
        description="Build a xorq expression and immediately run it"
    )
    parser.add_argument("script", help="Python file containing a xorq expression")
    parser.add_argument(
        "-e", "--expr-name", default="expr", help="Expression variable name (default: expr)"
    )
    parser.add_argument("-o", "--output", help="Output file path")
    parser.add_argument(
        "-f", "--format", help="Output format: parquet, csv, json, arrow (default: parquet)"
    )
    args = parser.parse_args()

    print(f"==> Building '{args.expr_name}' from {args.script}")
    result = subprocess.run(
        ["xorq", "build", args.script, "-e", args.expr_name],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        print(result.stderr, file=sys.stderr)
        sys.exit(result.returncode)

    build_path = result.stdout.strip()
    print(f"==> Running build at {build_path}")

    run_cmd = ["xorq", "run", build_path]
    if args.output:
        run_cmd += ["-o", args.output]
    if args.format:
        run_cmd += ["-f", args.format]

    sys.exit(subprocess.call(run_cmd))


if __name__ == "__main__":
    main()
