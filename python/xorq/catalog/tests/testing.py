"""Test primitives for the catalog TUI.

Three composable primitives that replace ad-hoc ``pilot.pause()`` calls:

- ``settle``     – wait until all workers are done and the message queue is drained.
- ``wait_until`` – settle in a loop until a predicate holds.
- ``run_script`` – execute a sequence of press/click/assert steps with auto-settle.
"""

from __future__ import annotations

import asyncio
from typing import Callable

from attr import frozen
from textual.pilot import Pilot


async def settle(pilot: Pilot, *, timeout: float = 5.0) -> None:
    """Wait until the app has no running workers and the message queue is empty.

    ``pilot.pause()`` drains messages but ignores ``@work(thread=True)``
    workers.  ``settle`` waits for both.
    """
    deadline = asyncio.get_running_loop().time() + timeout

    while True:
        await pilot.pause()

        active = [w for w in pilot.app.workers if w.is_running]
        if not active:
            await pilot.pause()
            return

        remaining = deadline - asyncio.get_running_loop().time()
        if remaining <= 0:
            names = [w.name for w in active]
            raise TimeoutError(f"Workers still running after {timeout}s: {names}")

        try:
            await asyncio.wait_for(
                asyncio.gather(*[w.wait() for w in active]),
                timeout=remaining,
            )
        except (asyncio.CancelledError, asyncio.TimeoutError):
            pass

        await pilot.pause()


async def wait_until(
    pilot: Pilot,
    predicate: Callable[[], bool],
    *,
    timeout: float = 5.0,
    poll: float = 0.05,
) -> None:
    """Wait until *predicate* returns truthy, settling between checks."""
    deadline = asyncio.get_running_loop().time() + timeout
    while True:
        remaining = max(0.1, deadline - asyncio.get_running_loop().time())
        await settle(pilot, timeout=remaining)
        if predicate():
            return
        if asyncio.get_running_loop().time() > deadline:
            raise TimeoutError(f"Condition not met after {timeout}s")
        await asyncio.sleep(poll)


# ---------------------------------------------------------------------------
# Scripted interaction steps
# ---------------------------------------------------------------------------


@frozen
class Step:
    """Base for scripted TUI interaction steps."""


@frozen
class Press(Step):
    keys: tuple[str, ...]


@frozen
class Click(Step):
    selector: str


@frozen
class Settle(Step):
    pass


@frozen
class Assert(Step):
    check: Callable[[Pilot], object]


@frozen
class WaitUntil(Step):
    predicate: Callable[[], bool]
    timeout: float = 5.0


async def run_script(pilot: Pilot, *steps: Step) -> None:
    """Execute a sequence of TUI interactions with automatic settling.

    Every ``Press`` / ``Click`` step settles before proceeding.  ``Assert``
    steps whose *check* returns a ``bool`` are verified; callables that
    perform their own assertions (returning ``None``) pass through.

    Usage::

        await run_script(pilot,
            Press(("j",)),
            Assert(lambda p: table.cursor_row == 1),
            Press(("k",)),
            Assert(lambda p: table.cursor_row == 0),
        )
    """
    for step in steps:
        match step:
            case Press(keys=keys):
                await pilot.press(*keys)
                await settle(pilot)
            case Click(selector=sel):
                await pilot.click(sel)
                await settle(pilot)
            case Settle():
                await settle(pilot)
            case Assert(check=check):
                match check(pilot):
                    case False:
                        raise AssertionError(f"Assert step failed: {check}")
                    case _:
                        pass
            case WaitUntil(predicate=pred, timeout=t):
                await wait_until(pilot, pred, timeout=t)
