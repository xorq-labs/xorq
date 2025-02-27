import argparse
import os
import sys

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


def run_command(builds_dir, hash_id):
    """
    Run a build, by recreating the expression from the build

    Parameters
    ----------
    builds_dir : Path to the builds directory
    hash_id : The hash identifier of the build to run

    Returns
    -------

    """
    try:
        build_manager = BuildManager(builds_dir)
        expr = build_manager.load_expr(hash_id)
        print(f"Executing expression with {hash_id}")
        expr.execute()

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


def main():
    """Main entry point for the xorq CLI."""
    parser = argparse.ArgumentParser(description="xorq - build and run expressions")

    # Create subparsers for different commands
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")
    subparsers.required = True

    # Create parser for the "build" command
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

    # Create parser for the "run" command
    run_parser = subparsers.add_parser(
        "run", help="Run a build from a builds directory"
    )
    run_parser.add_argument("builds_dir", help="Path to the builds directory")
    run_parser.add_argument(
        "expression_hash", help="Hash identifier of the build to run"
    )

    args = parser.parse_args()

    match args.command:
        case "build":
            expressions = [args.expressions] if args.expressions else []
            build_command(args.script_path, expressions, args.builds_dir)
        case "run":
            run_command(args.builds_dir, args.expression_hash)


if __name__ == "__main__":
    main()
