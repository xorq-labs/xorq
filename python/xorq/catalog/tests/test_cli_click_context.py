from __future__ import annotations

from collections.abc import Callable
from contextlib import AbstractContextManager

import click
import pytest
from click.testing import CliRunner, Result
from git import NoSuchPathError

from xorq.catalog.cli import click_context_catalog, click_context_default
from xorq.cli import PdbGroup


Handler = Callable[[click.Context], AbstractContextManager[None]]
Body = Callable[[click.Context], None]

HANDLERS = (
    pytest.param(click_context_catalog, id="catalog"),
    pytest.param(click_context_default, id="default"),
)


def invoke(handler: Handler, body: Body, *args: str) -> Result:
    @click.group(cls=PdbGroup)
    @click.option("--pdb", "use_pdb", is_flag=True)
    @click.option("--path")
    def group(use_pdb: bool, path: str | None) -> None:
        pass

    @group.command()
    @click.pass_context
    def c(ctx: click.Context) -> None:
        with handler(ctx):
            body(ctx)

    return CliRunner().invoke(group, [*args, "c"])


def ctx_exit(code: int) -> Body:
    return lambda ctx: ctx.exit(code)


def raising(exc: BaseException) -> Body:
    def body(ctx: click.Context) -> None:
        raise exc

    return body


@pytest.mark.parametrize("handler", HANDLERS)
@pytest.mark.parametrize(
    "code", (pytest.param(0, id="exit-0"), pytest.param(3, id="exit-3"))
)
def test_click_context_keeps_ctx_exit_code(handler: Handler, code: int) -> None:
    result = invoke(handler, ctx_exit(code))
    assert (result.exit_code, result.output) == (code, "")


@pytest.mark.parametrize("handler", HANDLERS)
def test_click_context_lets_abort_through(handler: Handler) -> None:
    result = invoke(handler, raising(click.Abort()))
    assert (result.exit_code, result.output) == (1, "Aborted!\n")


@pytest.mark.parametrize("handler", HANDLERS)
def test_click_context_passes_click_exception_unchanged(handler: Handler) -> None:
    result = invoke(handler, raising(click.UsageError("bad usage")))
    assert result.exit_code == 2
    assert "Error: bad usage" in result.output


@pytest.mark.parametrize("handler", HANDLERS)
def test_click_context_wraps_ordinary_exception(handler: Handler) -> None:
    result = invoke(handler, raising(ValueError("boom")))
    assert (result.exit_code, result.output) == (1, "Error: boom\n")


@pytest.mark.parametrize("handler", HANDLERS)
def test_click_context_pdb_reraises_ordinary_exception(
    handler: Handler, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen = []
    monkeypatch.setattr("xorq.cli.pdb.post_mortem", seen.append)
    result = invoke(handler, raising(ValueError("boom")), "--pdb")
    assert result.exit_code == 1
    assert len(seen) == 1
    assert "ValueError: boom" in result.output


def test_click_context_catalog_no_such_path_hints_init() -> None:
    result = invoke(
        click_context_catalog, raising(NoSuchPathError("/nope")), "--path", "/nope"
    )
    assert result.exit_code == 1
    assert "Run `xorq catalog --path /nope init` to create it." in result.output
