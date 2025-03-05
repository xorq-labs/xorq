import argparse
import os
import sys
from pathlib import Path

from xorq.common.utils.import_utils import import_from_path
from xorq.ibis_yaml.compiler import BuildManager
from xorq.vendor.ibis import Expr


def build_command(script_path, expression, builds_dir="builds"):
    """
    Generate artifacts from an expression in a given Python script

    Parameters
    ----------
    script_path : Path to the Python script
    expression : The name of the expression to build
    builds_dir : Directory where artifacts will be generated

    Returns
    -------

    """

    if len(expression) != 1:
        print("Expected one, and only one expression", file=sys.stderr)
        sys.exit(1)

    (expression,) = expression

    if not os.path.exists(script_path):
        print(f"Error: Script not found at {script_path}", file=sys.stderr)
        sys.exit(1)

    print(f"Building {expression} from {script_path}")

    build_manager = BuildManager(builds_dir)

    vars_module = import_from_path(script_path)

    if not hasattr(vars_module, expression):
        print(f"Expression {expression} not found", file=sys.stderr)
        sys.exit(1)

    expr = getattr(vars_module, expression)

    if not isinstance(expr, Expr):
        print(
            f"The object {expression} must be an instance of {type(expr)}",
            file=sys.stderr,
        )
        sys.exit(1)

    expr_hash = build_manager.compile_expr(expr)
    print(
        f"Written '{expression}' to {build_manager.artifact_store.get_path(expr_hash)}"
    )


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

    try:
        expr_path = Path(expr_path)
        build_manager = BuildManager(expr_path.parent)
        expr = build_manager.load_expr(expr_path.stem)

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
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


def main():
    """Main entry point for the xorq CLI."""
    parser = argparse.ArgumentParser(description="xorq - build and run expressions")

    subparsers = parser.add_subparsers(dest="command", help="Command to execute")
    subparsers.required = True

    build_parser = subparsers.add_parser(
        "build", help="Generate artifacts from an expression"
    )
    build_parser.add_argument("script_path", help="Path to the Python script")
    build_parser.add_argument(
        "-e",
        "--expressions",
        nargs="?",
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
        "--output-path",
        default=None,
        help=f"Path to write output (default: {os.devnull})",
    )
    run_parser.add_argument(
        "--format",
        choices=["csv", "json", "parquet"],
        default="parquet",
        help="Output format (default: parquet)",
    )

    args = parser.parse_args()

    match args.command:
        case "build":
            expressions = [args.expressions] if args.expressions else []
            build_command(args.script_path, expressions, args.builds_dir)
        case "run":
            run_command(args.build_path, args.output_path, args.format)
        case _:
            raise ValueError(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()
