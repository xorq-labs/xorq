import pathlib

from xorq.vendor.ibis.config import Config


class BuildConfig(Config):
    hash_length: int = 12
    _build_path: pathlib.Path = pathlib.Path.cwd() / "builds"

    @property
    def build_path(self) -> pathlib.Path:
        return self._build_path

    @build_path.setter
    def build_path(self, value: pathlib.Path):
        self._build_path = value


config = BuildConfig()
