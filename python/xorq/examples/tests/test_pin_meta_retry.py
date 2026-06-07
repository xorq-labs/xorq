from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from xorq.examples.core import _pin_meta_with_retry


def _make_board(side_effects: list) -> object:
    calls = iter(side_effects)

    class FakeBoard:
        def pin_meta(self, name: str) -> object:
            effect = next(calls)
            if isinstance(effect, Exception):
                raise effect
            return effect

    return FakeBoard()


@pytest.fixture
def mock_sleep() -> MagicMock:
    with patch("xorq.examples.core.time.sleep") as m:
        yield m


def test_success_first_attempt(mock_sleep: MagicMock) -> None:
    meta = SimpleNamespace(file="data.parquet")
    board = _make_board([meta])
    assert _pin_meta_with_retry(board, "test") is meta
    mock_sleep.assert_not_called()


def test_success_after_transient_exception(mock_sleep: MagicMock) -> None:
    meta = SimpleNamespace(file="data.parquet")
    board = _make_board([ConnectionError("timeout"), meta])
    assert _pin_meta_with_retry(board, "test") is meta
    mock_sleep.assert_called_once_with(1.0)


def test_success_after_none_then_exception(mock_sleep: MagicMock) -> None:
    meta = SimpleNamespace(file="data.parquet")
    board = _make_board([None, ValueError("transient"), meta])
    assert _pin_meta_with_retry(board, "test") is meta
    assert mock_sleep.call_count == 2
    mock_sleep.assert_any_call(1.0)
    mock_sleep.assert_any_call(2.0)


def test_all_exceptions_raises_with_cause(mock_sleep: MagicMock) -> None:
    original = ConnectionError("network down")
    board = _make_board([RuntimeError("a"), OSError("b"), original])
    with pytest.raises(RuntimeError, match="after 3 attempts") as exc_info:
        _pin_meta_with_retry(board, "test")
    assert exc_info.value.__cause__ is original


def test_all_none_raises_without_cause(mock_sleep: MagicMock) -> None:
    board = _make_board([None, None, None])
    with pytest.raises(RuntimeError, match="after 3 attempts") as exc_info:
        _pin_meta_with_retry(board, "test")
    assert exc_info.value.__cause__ is None


def test_exponential_backoff_delays(mock_sleep: MagicMock) -> None:
    board = _make_board([OSError(), OSError(), OSError()])
    with pytest.raises(RuntimeError):
        _pin_meta_with_retry(board, "test")
    assert mock_sleep.call_args_list == [
        ((1.0,),),
        ((2.0,),),
    ]
