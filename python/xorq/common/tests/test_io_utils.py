import subprocess

import pytest
import toolz

from xorq.common.utils.io_utils import Peeker


make_bytes_contains = toolz.flip(bytes.__contains__)


def test_peek_line_until():
    proc = subprocess.Popen(
        "sleep 1 && echo marker-text0 && sleep 1 && echo marker-text1",
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    peeker = Peeker(proc.stdout)

    condition = make_bytes_contains(b"marker-text1")
    peeker.peek_line_until(condition)
    proc.terminate()
    proc.wait()


def test_peek_line_until_with_timeout():
    proc = subprocess.Popen(
        "sleep 1 && echo marker-text0 && sleep 1 && echo marker-text1",
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    peeker = Peeker(proc.stdout)

    marker_text = b"marker-text1"
    condition = toolz.flip(bytes.__contains__)(marker_text)
    peeker.peek_line_until(condition, timeout=3)
    proc.terminate()
    proc.wait()


def test_peek_line_until_with_timeout_raises():
    proc = subprocess.Popen(
        "sleep 1 && echo marker-text0 && sleep 1 && echo marker-text1",
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    peeker = Peeker(proc.stdout)

    with pytest.raises(TimeoutError):
        marker_text = b"marker-text1"
        condition = toolz.flip(bytes.__contains__)(marker_text)
        peeker.peek_line_until(condition, timeout=1)
    proc.terminate()
    proc.wait()
