"""Timing breakdown for xorq uv-build and xorq uv-run.

Usage:
    python -m xorq.ibis_yaml.bench_packager build <script_path>
    python -m xorq.ibis_yaml.bench_packager run <build_path>
    python -m xorq.ibis_yaml.bench_packager load <build_path>
"""

import os
import time

import click


def _fmt(label, duration, total=None):
    pct = f" ({100 * duration / total:.0f}%)" if total else ""
    return f"  {label:30s} {duration:.3f}s{pct}"


def _print_table(title, rows, total):
    click.echo(f"\n{title}")
    click.echo("=" * 50)
    for label, duration in rows:
        click.echo(_fmt(label, duration, total))
    click.echo(f"  {'─' * 48}")
    click.echo(_fmt("Total", total))


@click.group()
def cli():
    """Timing breakdown for xorq packager pipeline."""


@cli.command()
@click.argument("script_path")
@click.option("--expr-name", "-e", default="expr")
@click.option("--builds-dir", default="builds")
@click.option(
    "--extra", "extras", multiple=True, help="Optional dependency group to include"
)
def build(script_path, expr_name, builds_dir, extras):
    """Time each step of xorq uv-build."""
    from xorq.ibis_yaml.packager import WheelPackager, uv_tool_run  # noqa: PLC0415

    t = [time.monotonic()]

    packager = WheelPackager.from_script_path(script_path, extras=extras)
    t.append(time.monotonic())

    _ = packager._wheel_path
    t.append(time.monotonic())

    _ = packager.requirements_path
    t.append(time.monotonic())

    result = uv_tool_run(
        "xorq",
        "build",
        script_path,
        "-e",
        expr_name,
        "--builds-dir",
        builds_dir,
        python_version=packager.python_version,
        with_=packager.wheel_path,
        with_requirements=packager.requirements_path,
        check=False,
    )
    t.append(time.monotonic())

    total = t[-1] - t[0]
    _print_table(
        f"xorq uv-build  {script_path}",
        [
            ("WheelPackager init", t[1] - t[0]),
            ("uv build --wheel", t[2] - t[1]),
            ("uv export (requirements)", t[3] - t[2]),
            ("uv tool run xorq build", t[4] - t[3]),
        ],
        total,
    )

    click.echo()
    if result.returncode:
        click.echo(f"  xorq build exited {result.returncode}", err=True)
        if result.stderr:
            for line in result.stderr.strip().splitlines()[-5:]:
                click.echo(f"    {line}", err=True)
    else:
        click.echo(f"  build_path: {result.stdout.strip()}")


@cli.command()
@click.argument("build_path")
def run(build_path):
    """Time each step of xorq uv-run."""
    from xorq.ibis_yaml.packager import PackagedRunner, uv_tool_run  # noqa: PLC0415

    t = [time.monotonic()]

    runner = PackagedRunner(build_path=build_path)
    t.append(time.monotonic())

    result = uv_tool_run(
        "xorq",
        "run",
        str(build_path),
        "--format",
        "parquet",
        python_version=runner.python_version,
        with_=runner.wheel_path,
        with_requirements=runner.requirements_path,
        capturing=False,
        check=False,
    )
    t.append(time.monotonic())

    total = t[-1] - t[0]
    _print_table(
        f"xorq uv-run  {build_path}",
        [
            ("PackagedRunner init", t[1] - t[0]),
            ("uv tool run xorq run", t[2] - t[1]),
        ],
        total,
    )

    if result.returncode:
        click.echo(f"\n  xorq run exited {result.returncode}", err=True)


@cli.command()
@click.argument("build_path")
def load(build_path):
    """Time load_expr internals (in-process, no uv tool run overhead)."""
    t = [time.monotonic()]

    from xorq.ibis_yaml.compiler import (  # noqa: PLC0415
        ArtifactStore,
        ExprLoader,
        YamlExpressionTranslator,
        _ensure_translate_registered,
        hydrate_cons,
    )
    from xorq.ibis_yaml.enums import DumpFiles  # noqa: PLC0415

    t.append(time.monotonic())

    artifact_store = ArtifactStore(build_path)
    profiles_dict = artifact_store.load_yaml(DumpFiles.profiles)
    profiles = hydrate_cons(profiles_dict, lazy=True)
    t.append(time.monotonic())

    yaml_dict = artifact_store.load_yaml(DumpFiles.expr)
    t.append(time.monotonic())

    _ensure_translate_registered()
    t.append(time.monotonic())

    expr = YamlExpressionTranslator.from_yaml(yaml_dict, profiles=profiles)
    t.append(time.monotonic())

    expr = ExprLoader(build_path).deferred_reads_to_memtables(expr, build_path)
    t.append(time.monotonic())

    from xorq.cli import arbitrate_output_format  # noqa: PLC0415

    arbitrate_output_format(expr, os.devnull, "parquet")
    t.append(time.monotonic())

    total = t[-1] - t[0]
    _print_table(
        f"load_expr breakdown  {build_path}",
        [
            ("import compiler + deps", t[1] - t[0]),
            ("hydrate_cons", t[2] - t[1]),
            ("load expr YAML", t[3] - t[2]),
            ("ensure_translate (sklearn)", t[4] - t[3]),
            ("from_yaml (translate)", t[5] - t[4]),
            ("deferred_reads_to_memtables", t[6] - t[5]),
            ("expr.execute()", t[7] - t[6]),
        ],
        total,
    )


if __name__ == "__main__":
    cli()
