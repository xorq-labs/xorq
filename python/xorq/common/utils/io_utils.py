from contextlib import contextmanager
from io import TextIOWrapper
from pathlib import Path
from urllib.parse import urlparse


@contextmanager
def maybe_open(obj, *args, **kwargs):
    if isinstance(obj, TextIOWrapper):
        yield obj
    else:
        with open(obj, *args, **kwargs) as fh:
            yield fh


def extract_suffix(path: str | Path) -> str:
    """
    Extracts the file extension (suffix) from a given file path or URL.

    Parameters
    ----------
    path : str | Path
        File path or URL to extract suffix from.

    Returns
    -------
    str
        The file extension (including the leading dot), or an empty string if there is none.

    """

    if isinstance(path, str):
        path = Path(urlparse(path).path)

    return path.suffix
