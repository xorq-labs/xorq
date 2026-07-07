#!/usr/bin/env python3
"""Generate the CLI reference pages from Click commands.

Walks the Click command tree in ``xorq.cli`` and ``xorq.catalog.cli`` and
writes one Quarto page per command plus an index page to
``docs/api_reference/cli/``. The output directory is gitignored; the docs
build regenerates it (see ``just docs-apigen``).

Page content comes from the command docstrings (Click ``help``), so the
docstrings are the single source of truth for both ``--help`` output and the
reference site. Two literal-block conventions are recognized inside
docstrings (mark the paragraph with ``\\b`` so Click does not rewrap it):

    \\b
    Arguments:
      NAME  Description of the positional argument.

    \\b
    Examples:
      # comment describing the invocation
      xorq command --flag value

``Arguments:`` blocks render as the page's Arguments section; ``Examples:``
blocks render as a bash fence. Any other paragraph passes through as
markdown prose.

Cross-page curation (section grouping, sidebar order, see-also links, and
redirect aliases) lives in the configuration tuples below. Every visible
command must appear in exactly one group — the script fails otherwise, so
adding a CLI command forces a conscious navigation decision.

Usage:
    uv run --no-sync python docs/generate_cli_reference.py
"""

import re
import shutil
import sys
from pathlib import Path

import click


DOCS_DIR = Path(__file__).parent
OUTPUT_DIR = DOCS_DIR / "api_reference" / "cli"
CATALOG_OUTPUT_DIR = OUTPUT_DIR / "catalog"

LITERAL_MARKER = "\b"

# =============================================================================
# Curated configuration
# =============================================================================

# (section title, (command name, ...)) per module. Drives the index page
# grouping and the sidebar order (emitted as `order:` frontmatter, consumed
# by the _quarto.yml glob for this directory).
MAIN_GROUPS = (
    (
        "Project setup",
        ("init", "completion", "install-completion"),
    ),
    (
        "Build",
        ("build", "uv build"),
    ),
    (
        "Run",
        (
            "run",
            "uv run",
            "run-cached",
            "uv run-cached",
            "run-unbound",
            "uv run-unbound",
        ),
    ),
    (
        "Serve",
        ("serve-flight-udxf", "serve-unbound"),
    ),
    (
        "Cache",
        ("pin", "unpin"),
    ),
)

CATALOG_GROUPS = (
    (
        "Repo lifecycle",
        ("init", "clone", "info", "default", "tui"),
    ),
    (
        "Entries and aliases",
        (
            "add",
            "remove",
            "list",
            "show",
            "schema",
            "get",
            "add-alias",
            "remove-alias",
            "list-aliases",
        ),
    ),
    (
        "Composition and execution",
        ("compose", "run", "run-cached", "serve-unbound"),
    ),
    (
        "Cache",
        ("pin", "unpin"),
    ),
    (
        "Sync, audit, replay",
        (
            "push",
            "pull",
            "sync",
            "set-remote",
            "embed-readonly",
            "check",
            "gc",
            "log",
            "replay",
        ),
    ),
)

# Page key -> markdown bullet lines for the See also section. Keys are the
# output path relative to the cli directory, without extension.
SEE_ALSO = {
    "completion": (
        "[`install-completion`](install-completion.qmd)—install the completion"
        " script to a standard location instead of evaluating it inline.",
    ),
    "install-completion": (
        "[`completion`](completion.qmd)—print the completion script to stdout"
        " instead of installing it.",
    ),
    "build": (
        "[`run`](run.qmd)—execute a build artifact",
        "[`uv build`](uv-build.qmd)—same semantics, inside a uv-managed environment",
        "[`catalog add`](catalog/add.qmd)—add a build to a catalog",
    ),
    "run": (
        "[`build`](build.qmd)—produce the build artifact this command executes",
        "[`uv run`](uv-run.qmd)—same semantics, inside a uv-managed environment",
        "[`run-cached`](run-cached.qmd)—same semantics, with a parquet cache wrapper",
    ),
    "run-cached": (
        "[`run`](run.qmd)—same semantics, without the cache wrapper",
        "[`uv run-cached`](uv-run-cached.qmd)—same semantics, inside a"
        " uv-managed environment",
    ),
    "run-unbound": (
        "[`serve-unbound`](serve-unbound.qmd)—serve the same unbound expression"
        " as a Flight endpoint",
        "[`uv run-unbound`](uv-run-unbound.qmd)—same semantics, inside a"
        " uv-managed environment",
    ),
    "serve-unbound": (
        "[`run-unbound`](run-unbound.qmd)—run the same unbound expression once"
        " over Arrow IPC instead of serving it",
        "[`catalog serve-unbound`](catalog/serve-unbound.qmd)—serve a catalog"
        " entry instead of a build directory",
    ),
    "serve-flight-udxf": (
        "[`serve-unbound`](serve-unbound.qmd)—serve an unbound expression"
        " instead of UDXF nodes",
    ),
    "uv-build": (
        "[`build`](build.qmd)—same semantics, in the current environment",
        "[`uv run`](uv-run.qmd)—execute the build in its packaged environment",
    ),
    "uv-run": (
        "[`run`](run.qmd)—same semantics, in the current environment",
        "[`uv build`](uv-build.qmd)—produce the packaged build this command executes",
    ),
    "uv-run-cached": (
        "[`run-cached`](run-cached.qmd)—same semantics, in the current environment",
    ),
    "uv-run-unbound": (
        "[`run-unbound`](run-unbound.qmd)—same semantics, in the current environment",
    ),
    "catalog/add": (
        "[`catalog list`](list.qmd)—list catalog entries",
        "[`catalog add-alias`](add-alias.qmd)—attach an alias to an existing entry",
        "[`catalog remove`](remove.qmd)—remove an entry from the catalog",
    ),
    "catalog/remove": (
        "[`catalog remove-alias`](remove-alias.qmd)—remove an alias without"
        " removing the underlying entry",
        "[`catalog list`](list.qmd)—list catalog entries",
    ),
    "catalog/add-alias": (
        "[`catalog list-aliases`](list-aliases.qmd)—list registered aliases",
        "[`catalog remove-alias`](remove-alias.qmd)—remove an alias",
    ),
    "catalog/remove-alias": (
        "[`catalog remove`](remove.qmd)—remove an entry itself",
        "[`catalog list-aliases`](list-aliases.qmd)—list registered aliases",
    ),
    "catalog/list": (
        "[`catalog list-aliases`](list-aliases.qmd)—list registered aliases",
        "[`catalog show`](show.qmd)—show full metadata for an entry",
        "[`catalog schema`](schema.qmd)—show schema for an entry",
    ),
    "catalog/list-aliases": (
        "[`catalog list`](list.qmd)—list catalog entries",
        "[`catalog add-alias`](add-alias.qmd)—add an alias to an entry",
    ),
    "catalog/show": (
        "[`catalog schema`](schema.qmd)—schema-only view",
        "[`catalog list`](list.qmd)—list entries",
    ),
    "catalog/schema": (
        "[`catalog show`](show.qmd)—full metadata for an entry",
        "[`catalog list`](list.qmd)—list available entries",
    ),
    "catalog/get": (
        "[`catalog add`](add.qmd)—re-add an exported archive to another catalog",
    ),
    "catalog/init": (
        "[`catalog clone`](clone.qmd)—clone an existing catalog from a URL",
        "[`catalog default`](default.qmd)—manage the default catalog name",
    ),
    "catalog/clone": (
        "[`catalog init`](init.qmd)—create a fresh catalog instead of cloning",
    ),
    "catalog/info": (
        "[`catalog show`](show.qmd)—per-entry metadata",
        "[`catalog schema`](schema.qmd)—per-entry schemas",
    ),
    "catalog/push": (
        "[`catalog pull`](pull.qmd)—pull from remotes",
        "[`catalog sync`](sync.qmd)—pull then push",
    ),
    "catalog/pull": (
        "[`catalog push`](push.qmd)—push to remotes",
        "[`catalog sync`](sync.qmd)—pull then push",
    ),
    "catalog/sync": (
        "[`catalog pull`](pull.qmd)—pull from remotes",
        "[`catalog push`](push.qmd)—push to remotes",
    ),
    "catalog/set-remote": (
        "[`catalog push`](push.qmd)—push commits and annex content to the remote",
        "[`catalog pull`](pull.qmd)—pull from the remote",
        "[`catalog clone`](clone.qmd)—clone a catalog from a remote URL",
    ),
    "catalog/compose": (
        "[`catalog run`](run.qmd)—execute a composed expression for data output",
    ),
    "catalog/run": (
        "[`catalog compose`](compose.qmd)—persist a composed expression instead"
        " of executing it",
        "[`catalog run-cached`](run-cached.qmd)—same semantics, but wraps the"
        " result in a parquet cache",
    ),
    "catalog/run-cached": (
        "[`catalog run`](run.qmd)—same semantics, without the cache wrapper",
        "[`run-cached`](../run-cached.qmd)—cache strategies explained",
    ),
    "catalog/serve-unbound": (
        "[`serve-unbound`](../serve-unbound.qmd)—serve an unbound expression"
        " from a build directory",
        "[`serve-flight-udxf`](../serve-flight-udxf.qmd)—serve a build as a"
        " Flight UDXF",
        "[`catalog run`](run.qmd)—compose and execute catalog entries locally",
    ),
    "catalog/log": (
        "[`catalog replay`](replay.qmd)—replay this history into a different catalog",
    ),
    "catalog/replay": (
        "[`catalog log`](log.qmd)—view the operation history to replay",
    ),
}

# Page key -> redirect aliases preserved from renamed pages.
FRONTMATTER_ALIASES = {
    "catalog/list": ("/api_reference/cli/catalog/ls.html",),
    "catalog/remove": ("/api_reference/cli/catalog/rm.html",),
}

INDEX_INTRO = """\
Xorq provides command-line tools for building, running, serving, and cataloging
your data and ML pipelines."""


# =============================================================================
# Click tree traversal
# =============================================================================


def iter_visible_commands(group, known_groups=()):
    """Yield (name, command) for the group's visible terminal commands.

    Nested groups the generator doesn't know how to flatten are a hard
    error — pass their names via *known_groups* once handled.
    """
    for name, cmd in group.commands.items():
        if cmd.hidden:
            continue
        if isinstance(cmd, click.Group):
            if name not in known_groups:
                raise SystemExit(
                    f"nested command group not handled by the generator: {name!r}"
                    " — teach load_*_commands() to flatten it"
                    " (see how the 'uv' group is handled)"
                )
            continue
        yield (name, cmd)


def load_main_commands():
    """Return {key: (cmd, invocation)} for xorq.cli, flattening the uv group."""
    from xorq.cli import cli as main_cli  # noqa: PLC0415

    commands = {}
    for name, cmd in iter_visible_commands(main_cli, known_groups=("uv",)):
        commands[name] = (cmd, f"xorq {name}")
    uv_group = main_cli.commands.get("uv")
    if uv_group is not None:
        for name, cmd in iter_visible_commands(uv_group):
            commands[f"uv {name}"] = (cmd, f"xorq uv {name}")
    return commands


def load_catalog_commands():
    """Return ({name: (cmd, invocation)}, group) for xorq.catalog.cli."""
    from xorq.catalog.cli import cli as catalog_cli  # noqa: PLC0415

    commands = {
        name: (cmd, f"xorq catalog {name}")
        for name, cmd in iter_visible_commands(catalog_cli)
    }
    return (commands, catalog_cli)


def validate_groups(groups, commands, label):
    """Fail unless the group config and the Click tree match exactly."""
    configured = tuple(name for (_, names) in groups for name in names)
    duplicates = tuple(name for name in set(configured) if configured.count(name) > 1)
    missing = tuple(name for name in commands if name not in configured)
    stale = tuple(name for name in configured if name not in commands)
    problems = (
        *(f"listed twice in {label} groups: {name}" for name in duplicates),
        *(
            f"command not in {label} groups (add it to a section): {name}"
            for name in missing
        ),
        *(f"stale entry in {label} groups (no such command): {name}" for name in stale),
    )
    if problems:
        raise SystemExit("\n".join(("group config mismatch:", *problems)))


# =============================================================================
# Docstring parsing
# =============================================================================


def split_paragraphs(text):
    """Split help text into (is_literal, lines) paragraphs, Click-style."""
    paragraphs = []
    current = []
    literal = False
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            if current:
                paragraphs.append((literal, tuple(current)))
                current = []
                literal = False
        elif stripped == LITERAL_MARKER:
            literal = True
        else:
            current.append(line)
    if current:
        paragraphs.append((literal, tuple(current)))
    return tuple(paragraphs)


def parse_block_items(lines):
    """Parse ``NAME  description`` lines (with continuations) into pairs."""
    items = []
    for line in lines:
        stripped = line.strip()
        (name, _, rest) = stripped.partition("  ")
        if rest and not line.startswith((" " * 8,)):
            items.append((name, rest.strip()))
        elif items:
            # continuation line: append to the previous description
            (prev_name, prev_rest) = items[-1]
            items[-1] = (prev_name, f"{prev_rest} {stripped}")
    return tuple(items)


def render_prose_paragraph(lines):
    """Render a docstring paragraph as markdown prose.

    Strips per-line indentation (defensive: an un-dedented ``cmd.help``
    would otherwise render as an indented code block) and inserts the
    blank line markdown needs between a trailing-colon header line and
    the list items that follow it (e.g. the ``Templates:`` block).
    """
    stripped = tuple(line.strip() for line in lines)
    if (
        len(stripped) > 1
        and stripped[0].endswith(":")
        and re.match(r"(-|\d+\.)\s", stripped[1])
    ):
        return "\n".join((stripped[0], "", *stripped[1:]))
    return "\n".join(stripped)


def parse_docstring(help_text):
    """Split a command's help into (prose, argument items, example lines)."""
    prose = []
    arguments = []
    examples = []
    for is_literal, lines in split_paragraphs(help_text or ""):
        header = lines[0].strip().lower() if lines else ""
        if is_literal and header == "arguments:":
            arguments.extend(parse_block_items(lines[1:]))
        elif is_literal and header == "examples:":
            examples.extend(line.strip() for line in lines[1:])
        else:
            prose.append(render_prose_paragraph(lines))
    return (tuple(prose), tuple(arguments), tuple(examples))


# =============================================================================
# Page rendering
# =============================================================================


def format_default(param):
    """Render an option's default for the options table."""
    if isinstance(param.show_default, str):
        return param.show_default
    default = param.default
    if param.secondary_opts:
        return f"`{param.opts[0]}`" if default else f"`{param.secondary_opts[0]}`"
    if default is None or default == () or "Sentinel" in repr(default):
        return "none"
    if isinstance(default, bool):
        return "on" if default else "off"
    if callable(default):
        return "(dynamic)"
    return f"`{default}`"


def format_option_names(param):
    """Render an option's names, including any off switch."""
    names = ", ".join(f"`{opt}`" for opt in param.opts)
    if param.secondary_opts:
        secondary = ", ".join(f"`{opt}`" for opt in param.secondary_opts)
        return f"{names} / {secondary}"
    return names


def iter_documented_options(cmd):
    """Yield the command's table-worthy options."""
    ctx = click.Context(cmd)
    for param in cmd.get_params(ctx):
        if isinstance(param, click.Argument):
            continue
        if param.name == "help" or getattr(param, "hidden", False):
            continue
        yield param


def render_usage(cmd, invocation):
    """Render the usage line with ``[OPTIONS]`` moved after the arguments."""
    ctx = click.Context(cmd)
    pieces = tuple(
        piece for piece in cmd.collect_usage_pieces(ctx) if piece != "[OPTIONS]"
    )
    has_options = any(True for _ in iter_documented_options(cmd))
    suffix = (*pieces, "[OPTIONS]") if has_options else pieces
    return " ".join((invocation, *suffix)).strip()


def render_options_table(cmd):
    """Render the options table, or '' for commands without options."""
    rows = tuple(
        f"| {format_option_names(param)} | {format_default(param)} |"
        f" {param.help or ''} |"
        for param in iter_documented_options(cmd)
    )
    if not rows:
        return ""
    header = ("| Option | Default | Description |", "|---|---|---|")
    return "\n".join((*header, *rows))


def render_frontmatter(title, order, aliases=()):
    # YAML single-quoted scalar: escape embedded quotes by doubling them.
    quoted = title.replace("'", "''")
    lines = ["---", f"title: '{quoted}'", f"order: {order}"]
    if aliases:
        lines.append("aliases:")
        lines.extend(f"  - {alias}" for alias in aliases)
    lines.append("---")
    return "\n".join(lines)


def render_page(key, cmd, invocation, order):
    """Render a command page as qmd."""
    (prose, arguments, examples) = parse_docstring(cmd.help)
    title = invocation.removeprefix("xorq ")
    sections = [render_frontmatter(title, order, FRONTMATTER_ALIASES.get(key, ()))]
    sections.extend(prose)
    sections.append(f"## Usage\n\n```bash\n{render_usage(cmd, invocation)}\n```")
    if arguments:
        items = "\n".join(
            f"- **`{name}`**—{description}" for (name, description) in arguments
        )
        sections.append(f"## Arguments\n\n{items}")
    options_table = render_options_table(cmd)
    if options_table:
        sections.append(f"## Options\n\n{options_table}")
    if examples:
        body = "\n".join(examples)
        sections.append(f"## Examples\n\n```bash\n{body}\n```")
    see_also = SEE_ALSO.get(key)
    if see_also:
        items = "\n".join(f"- {line}" for line in see_also)
        sections.append(f"## See also\n\n{items}")
    return "\n\n".join(sections) + "\n"


def render_index_table(groups, commands, href_prefix=""):
    """Render the grouped command tables for the index page."""
    sections = []
    for title, names in groups:
        rows = tuple(
            f"| [`{invocation.removeprefix('xorq ')}`]"
            f"({href_prefix}{filename_for(name)}) |"
            f" {cmd.get_short_help_str(limit=120)} |"
            for name in names
            for (cmd, invocation) in (commands[name],)
        )
        table = "\n".join(("| Command | Description |", "|---|---|", *rows))
        sections.append(f"### {title}\n\n{table}")
    return sections


def render_index(main_commands, catalog_commands, catalog_group):
    """Render the index page with grouped command tables."""
    sections = [render_frontmatter("Overview", 0), INDEX_INTRO]
    sections.append("## Commands")
    sections.extend(render_index_table(MAIN_GROUPS, main_commands))
    sections.append("## Catalog")
    (catalog_prose, _, _) = parse_docstring(catalog_group.help)
    sections.extend(catalog_prose)
    group_options = render_options_table(catalog_group)
    if group_options:
        sections.append(group_options)
    sections.extend(
        render_index_table(CATALOG_GROUPS, catalog_commands, href_prefix="catalog/")
    )
    return "\n\n".join(sections) + "\n"


# =============================================================================
# Output
# =============================================================================


def filename_for(name):
    return name.replace(" ", "-") + ".qmd"


def write_pages():
    main_commands = load_main_commands()
    (catalog_commands, catalog_group) = load_catalog_commands()
    validate_groups(MAIN_GROUPS, main_commands, "main")
    validate_groups(CATALOG_GROUPS, catalog_commands, "catalog")

    if OUTPUT_DIR.exists():
        shutil.rmtree(OUTPUT_DIR)
    CATALOG_OUTPUT_DIR.mkdir(parents=True)

    written = []
    order = 0
    for _, names in MAIN_GROUPS:
        for name in names:
            order += 1
            (cmd, invocation) = main_commands[name]
            key = filename_for(name).removesuffix(".qmd")
            path = OUTPUT_DIR / filename_for(name)
            path.write_text(render_page(key, cmd, invocation, order))
            written.append(path)
    order = 0
    for _, names in CATALOG_GROUPS:
        for name in names:
            order += 1
            (cmd, invocation) = catalog_commands[name]
            key = f"catalog/{filename_for(name).removesuffix('.qmd')}"
            path = CATALOG_OUTPUT_DIR / filename_for(name)
            path.write_text(render_page(key, cmd, invocation, order))
            written.append(path)
    index_path = OUTPUT_DIR / "index.qmd"
    index_path.write_text(render_index(main_commands, catalog_commands, catalog_group))
    written.append(index_path)
    return written


def main():
    written = write_pages()
    print(f"generated {len(written)} pages under {OUTPUT_DIR.relative_to(DOCS_DIR)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
