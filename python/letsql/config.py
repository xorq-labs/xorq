import pathlib
from typing import Union

from ibis.config import Config


class Options(Config):
    """LETSQL configuration options

    Attributes
    ----------

    cache_default_path : str

    """

    cache_default_path: Union[str, pathlib.Path] = pathlib.Path(
        "~/.local/share/letsql"
    ).expanduser()


options = Options()
