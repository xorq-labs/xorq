import pytest
import toolz

from xorq.common.utils.process_utils import Popened


make_bytes_contains = toolz.flip(bytes.__contains__)


def test_peek_line_until():
    popened = Popened(
        "sleep 1 && echo marker-text0 && sleep 1 && echo marker-text1",
        kwargs_tuple=(("shell", True),),
    )

    condition = make_bytes_contains(b"marker-text1")
    popened.stdout_peeker.peek_line_until(condition)


def test_peek_line_until_with_timeout():
    popened = Popened(
        "sleep 1 && echo marker-text0 && sleep 1 && echo marker-text1",
        kwargs_tuple=(("shell", True),),
    )

    marker_text = b"marker-text1"
    condition = toolz.flip(bytes.__contains__)(marker_text)
    popened.stdout_peeker.peek_line_until(condition, timeout=3)


def test_peek_line_until_with_timeout_raises():
    popened = Popened(
        "sleep 1 && echo marker-text0 && sleep 1 && echo marker-text1",
        kwargs_tuple=(("shell", True),),
    )

    with pytest.raises(TimeoutError):
        marker_text = b"marker-text1"
        condition = toolz.flip(bytes.__contains__)(marker_text)
        popened.stdout_peeker.peek_line_until(condition, timeout=1)
