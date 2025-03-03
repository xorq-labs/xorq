import argparse
import os
import sys

from xorq.common.utils.import_utils import import_from_path
from xorq.ibis_yaml.compiler import BuildManager
from xorq.vendor.ibis import Expr


def build_command(script_path, expression, target_dir="build"):
    """
    Generate artifacts from an expression in a given Python script

    Parameters
    ----------
    script_path : Path to the Python script
    expression : The name of the expression to build
    target_dir : Directory where artifacts will be generated

    Returns
    -------

    """

    if len(expression) > 1 or len(expression) == 0:
        print("Expected one, and only one expression", file=sys.stderr)
        sys.exit(1)

    (expression,) = expression

    if not os.path.exists(script_path):
        print(f"Error: Script not found at {script_path}", file=sys.stderr)
        sys.exit(1)

    print(f"Building {expression} from {script_path}")

    build_manager = BuildManager(target_dir)

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
        "--target-dir", default="build", help="Directory for all generated artifacts"
    )

    # Parse the arguments
    args = parser.parse_args()

    # Execute the appropriate command
    if args.command == "build":
        expressions = [args.expressions] if args.expressions else []
        build_command(args.script_path, expressions, args.target_dir)


if __name__ == "__main__":
    main()
