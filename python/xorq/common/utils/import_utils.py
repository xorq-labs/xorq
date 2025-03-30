import importlib
import json
import sys
import tempfile
import urllib
from pathlib import Path


def import_python(path, module_name=None):
    path = Path(path)
    return importlib.machinery.SourceFileLoader(
        module_name or path.stem, str(path)
    ).load_module()


def import_ipynb(path, module_name):
    """Import a Jupyter notebook as a module."""

    # Create a new module
    module = type(sys)(module_name)
    module.__file__ = str(path.resolve())

    # Add the module to sys.modules
    sys.modules[module_name] = module

    # Parse the notebook JSON
    with path.open("r", encoding="utf-8") as f:
        notebook = json.load(f)

    # Extract code from code cells
    code_cells = [
        cell for cell in notebook.get("cells", []) if cell.get("cell_type") == "code"
    ]

    # Combine all code cells, skipping cells marked with "# skip-import" comment
    code = ""
    for cell in code_cells:
        cell_source = "".join(cell.get("source", []))
        if "# skip-import" not in cell_source:
            code += cell_source + "\n\n"

    # Execute the code in the module's namespace
    try:
        exec(code, module.__dict__)
    except Exception as e:
        # Clean up if execution fails
        if module_name in sys.modules:
            del sys.modules[module_name]
        raise ImportError(f"Error executing notebook code: {e}")

    return module


def import_from_path(path, module_name="__main__"):
    """
    Import a Python script or Jupyter notebook as a module.

    Args:
        path (Path or str): pathlib.Path object or string pointing to a Python script (.py) or
                           Jupyter notebook (.ipynb)

    Returns:
        module: The imported module

    Raises:
        ImportError: If the file cannot be imported
        ValueError: If the file type is not supported
    """
    path = Path(path)

    # Check if path exists
    if not path.exists():
        raise ImportError(f"File not found: {path}")

    # Handle based on file extension
    if path.suffix == ".py":
        return import_python(path, module_name)
    elif path.suffix == ".ipynb":
        return import_ipynb(path, module_name)
    else:
        raise ValueError(
            f"Unsupported file type: {path.suffix}. Only .py and .ipynb files are supported."
        )


def import_from_gist(user, gist):
    path = f"https://gist.githubusercontent.com/{user}/{gist}/raw/"
    req = urllib.request.Request(path, method="GET")
    resp = urllib.request.urlopen(req)
    if resp.code != 200:
        raise ValueError
    with tempfile.NamedTemporaryFile() as ntfh:
        path = Path(ntfh.name)
        path.write_text(resp.read().decode("ascii"))
        module = import_python(path)
        return module
