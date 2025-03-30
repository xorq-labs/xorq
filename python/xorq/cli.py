import argparse
import os
import pdb
import sys
import traceback
from pathlib import Path

from xorq.common.utils.import_utils import import_from_path
from xorq.ibis_yaml.compiler import BuildManager
from xorq.vendor.ibis import Expr


def build_command(script_path, expr_name, builds_dir="builds"):
    """
    Generate artifacts from an expression in a given Python script

    Parameters
    ----------
    script_path : Path to the Python script
    expr_name : The name of the expression to build
    builds_dir : Directory where artifacts will be generated

    Returns
    -------

    """

    if not os.path.exists(script_path):
        raise ValueError(f"Error: Script not found at {script_path}")

    print(f"Building {expr_name} from {script_path}", file=sys.stderr)

    build_manager = BuildManager(builds_dir)

    vars_module = import_from_path(script_path)

    if not hasattr(vars_module, expr_name):
        raise ValueError(f"Expression {expr_name} not found")

    expr = getattr(vars_module, expr_name)

    if not isinstance(expr, Expr):
        raise ValueError(
            f"The object {expr_name} must be an instance of {Expr.__module__}.{Expr.__name__}"
        )

    expr_hash = build_manager.compile_expr(expr)
    print(
        f"Written '{expr_name}' to {build_manager.artifact_store.get_path(expr_hash)}",
        file=sys.stderr,
    )
    print(build_manager.artifact_store.get_path(expr_hash))


def run_command(expr_path, output_path=None, output_format="parquet"):
    """
    Execute an artifact

    Parameters
    ----------
    expr_path : str
        Path to the expr in the builds dir
    output_path : str
        Path to write output. Defaults to os.devnull
    output_format : str, optional
        Output format, either "csv", "json", or "parquet". Defaults to "parquet"

    Returns
    -------

    """
    if output_path is None:
        output_path = os.devnull

    expr_path = Path(expr_path)
    build_manager = BuildManager(expr_path.parent)
    expr = build_manager.load_expr(expr_path.stem)

    try:
        match output_format:
            case "csv":
                expr.to_csv(output_path)
            case "json":
                expr.to_json(output_path)
            case "parquet":
                expr.to_parquet(output_path)
            case _:
                raise ValueError(f"Unknown output_format: {output_format}")
    except Exception as e:
        raise RuntimeError(f"Error: {e}")


def parse_args(override=None):
    parser = argparse.ArgumentParser(description="xorq - build and run expressions")
    parser.add_argument("--pdb", action="store_true", help="Drop into pdb on failure")

    subparsers = parser.add_subparsers(dest="command", help="Command to execute")
    subparsers.required = True

    build_parser = subparsers.add_parser(
        "build", help="Generate artifacts from an expression"
    )
    build_parser.add_argument("script_path", help="Path to the Python script")
    build_parser.add_argument(
        "-e",
        "--expr-name",
        default="expr",
        help="Name of the expression variable in the Python script",
    )
    build_parser.add_argument(
        "--builds-dir", default="builds", help="Directory for all generated artifacts"
    )

    run_parser = subparsers.add_parser(
        "run", help="Run a build from a builds directory"
    )
    run_parser.add_argument("build_path", help="Path to the build script")
    run_parser.add_argument(
        "-o",
        "--output-path",
        default=None,
        help=f"Path to write output (default: {os.devnull})",
    )
    run_parser.add_argument(
        "-f",
        "--format",
        choices=["csv", "json", "parquet"],
        default="parquet",
        help="Output format (default: parquet)",
    )

    args = parser.parse_args(override)
    if getattr(args, "output_path", None) == "-":
        if args.format == "json":
            # FIXME: deal with windows
            args.output_path = sys.stdout
        else:
            args.output_path = sys.stdout.buffer
    return args


def main():
    """Main entry point for the xorq CLI."""
    args = parse_args()

    try:
        match args.command:
            case "build":
                build_command(args.script_path, args.expr_name, args.builds_dir)
            case "run":
                run_command(args.build_path, args.output_path, args.format)
            case _:
                raise ValueError(f"Unknown command: {args.command}")
    except Exception as e:
        if args.pdb:
            traceback.print_exception(e)
            pdb.post_mortem(e.__traceback__)
        else:
            print(e, file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
