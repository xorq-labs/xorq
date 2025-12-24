"""CLI I/O utilities for xorq."""

# Import main from parent cli module for backward compatibility
# The entry point expects xorq.cli:main
from pathlib import Path

from xorq.cli.io import read_arrow_stream, write_arrow_stream


# Get the cli.py module from parent directory
cli_py_path = Path(__file__).parent.parent / "cli.py"
if cli_py_path.exists():
    import importlib.util

    spec = importlib.util.spec_from_file_location("_xorq_cli_main", cli_py_path)
    _cli_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(_cli_module)
    main = _cli_module.main
else:
    # Fallback if structure changes
    from ..cli import main


__all__ = ["read_arrow_stream", "write_arrow_stream", "main"]
