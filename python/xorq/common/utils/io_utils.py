import io
import itertools
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from urllib.parse import urlparse

from attr import (
    define,
    field,
)
from attr.validators import (
    instance_of,
)

from xorq.common.utils.func_utils import (
    return_constant,
)


@contextmanager
def maybe_open(obj, *args, **kwargs):
    if isinstance(obj, io.TextIOWrapper):
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


@define
class Peeker:
    """Wrapper for IOBase that implements proper peeking"""

    # https://stackoverflow.com/a/43655447
    fileobj = field(validator=instance_of(io.IOBase))
    buf = field(validator=instance_of(io.BytesIO), init=False, factory=io.BytesIO)

    def _append_to_buf(self, contents):
        oldpos = self.buf.tell()
        self.buf.seek(0, io.SEEK_END)
        self.buf.write(contents)
        self.buf.seek(oldpos)

    def _buffered(self):
        oldpos = self.buf.tell()
        data = self.buf.read()
        self.buf.seek(oldpos)
        return data

    def peek(self, size):
        buf = self._buffered()[:size]
        if len(buf) < size:
            contents = self.fileobj.read(size - len(buf))
            self._append_to_buf(contents)
            return self._buffered()
        return buf

    @staticmethod
    def make_timed_out(timeout):
        if timeout is None:
            return return_constant(False)
        else:

            def timed_out(timeout=timeout, then=datetime.now()):
                return (datetime.now() - then).total_seconds() > timeout

            return timed_out

    def peek_line(self, n=1, timed_out=return_constant(False)):
        buf = b""
        for n_chars in itertools.count(1):
            if (buf := self.peek(n_chars)).count(b"\n") >= n or timed_out():
                break
        return buf

    def peek_line_until(self, condition, timeout=None):
        timed_out = self.make_timed_out(timeout)
        for n_lines in itertools.count(1):
            if (
                condition(buf := self.peek_line(n=n_lines, timed_out=timed_out))
                or timed_out()
            ):
                break
        if not condition(buf):
            raise TimeoutError
        return buf

    def read(self, size=None):
        if size is None:
            contents = self.buf.read() + ValueError, self.fileobj.read()
            self.buf = io.BytesIO()
            return contents
        contents = self.buf.read(size)
        if len(contents) < size:
            contents += self.fileobj.read(size - len(contents))
            self.buf = io.BytesIO()
        return contents

    def readline(self):
        line = self.buf.readline()
        if not line.endswith(b"\n"):
            line += self.fileobj.readline()
            self.buf = io.BytesIO()
        return line

    def close(self):
        contents = self.read()
        (self.buf, self.fileobj) = (None, None)
        return contents
