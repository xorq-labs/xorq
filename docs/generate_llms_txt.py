#!/usr/bin/env python3
"""Generate llms.txt and llms-full.txt from the docs configuration.

Builds the two files described by the llms.txt convention
(https://llmstxt.org/) and writes them to the docs root, where the Quarto
build copies them into the published site (see ``project.resources`` in
``_quarto.yml``). Both files are gitignored; the docs build regenerates them
(see ``just docs-apigen``).

- ``llms.txt`` is a compact index of the whole site: every API item from the
  ``quartodoc`` sections with its one-line docstring summary, every CLI
  command with its Click short help, and every guide page linked from the
  sidebar with a one-line description.
- ``llms-full.txt`` is the comprehensive version: full signatures and
  docstrings for the API surface, the complete ``--help`` text for every CLI
  command, and the full prose of every guide page with its YAML frontmatter
  stripped.

The ``quartodoc.sections`` block in ``_quarto.yml`` is the source of truth
for the API surface and the ``website.sidebar`` block for the guide pages, so
both files stay in sync with the rendered site without their own
configuration. Items are imported at generation time to extract docstrings
and signatures, so the package must be importable.

Objects vendored from ibis (``xorq.vendor.*`` and ``ibis``) are summarized as
signature plus first docstring paragraph: their full docstrings carry
interactive ibis-flavored examples that bloat the output and sometimes show
non-xorq idioms. Objects authored in xorq keep their full docstrings.

Usage:
    uv run --no-sync python docs/generate_llms_txt.py
"""

import importlib
import inspect
import re
import sys
from pathlib import Path

import click
import tomllib
import yaml


DOCS_DIR = Path(__file__).parent
REPO_ROOT = DOCS_DIR.parent

sys.path.insert(0, str(DOCS_DIR))

import generate_cli_reference as cli_ref  # noqa: E402


SITE_URL = "https://docs.xorq.dev/"
SEP_LINE = "-" * 70

# Resolved package prefixes treated as vendored upstream code: summarized
# instead of included verbatim (see module docstring).
VENDORED_PREFIXES = ("xorq.vendor", "ibis")


# =============================================================================
# Configuration loading
# =============================================================================


def read_quarto_config():
    with open(DOCS_DIR / "_quarto.yml") as f:
        return yaml.safe_load(f)


def read_package_metadata():
    """Return (name, description) from pyproject.toml."""
    with open(REPO_ROOT / "pyproject.toml", "rb") as f:
        project = tomllib.load(f)["project"]
    return (project["name"], project.get("description", ""))


def iter_api_sections(config):
    """Yield (title, desc, entries) per quartodoc section.

    Each entry is ``("item", name, package, include_inherited)`` for a
    documented object or ``("page", path, summary_name, summary_desc,
    subentries)`` for a ``kind: page`` group (e.g. the Type System pages),
    where subentries are item tuples.
    """

    def item_entry(item, default_package):
        if isinstance(item, str):
            return ("item", item, default_package, False)
        return (
            "item",
            item["name"],
            item.get("package") or default_package,
            bool(item.get("include_inherited")),
        )

    quartodoc = config["quartodoc"]
    default_package = quartodoc["package"]
    for section in quartodoc["sections"]:
        section_package = section.get("package") or default_package
        entries = []
        for item in section.get("contents", ()):
            if isinstance(item, dict) and item.get("kind") == "page":
                page_package = item.get("package") or section_package
                summary = item.get("summary", {})
                subentries = tuple(
                    item_entry(sub, page_package) for sub in item.get("contents", ())
                )
                entries.append(
                    (
                        "page",
                        item["path"],
                        summary.get("name", item["path"]),
                        summary.get("desc", ""),
                        subentries,
                    )
                )
            else:
                entries.append(item_entry(item, section_package))
        yield (section["title"], section.get("desc") or "", tuple(entries))


def iter_sidebar_pages(config):
    """Yield (group_title, section_title, href, link_text) from the sidebar.

    The sidebar is the curated source of truth for guide pages: orphan qmd
    files not linked there are deliberately excluded. ``auto:`` globs (the
    CLI pages) and the quartodoc-generated ``reference/`` pages are skipped —
    both surfaces get their own dedicated sections.
    """

    def walk(contents, group_title, section_title):
        for entry in contents:
            if isinstance(entry, str):
                yield (group_title, section_title, entry, None)
            elif "section" in entry:
                yield from walk(
                    entry.get("contents", ()), group_title, entry["section"]
                )
            elif "href" in entry:
                yield (group_title, section_title, entry["href"], entry.get("text"))

    for group in config["website"]["sidebar"]:
        for page in walk(group.get("contents", ()), group.get("title", ""), None):
            if not page[2].startswith("reference/"):
                yield page


# =============================================================================
# Object introspection
# =============================================================================


def resolve_item(package, name):
    """Import *package* and resolve the (possibly dotted) attribute *name*.

    A stale quartodoc entry (renamed or removed item) is a hard error, same
    as the group-config validation in generate_cli_reference.py.
    """
    try:
        obj = importlib.import_module(package)
        for part in name.split("."):
            obj = getattr(obj, part)
    except (ImportError, AttributeError) as e:
        raise SystemExit(
            f"stale quartodoc entry: cannot resolve {package}.{name}: {e}"
        ) from e
    return obj


def is_vendored(package):
    return package.startswith(VENDORED_PREFIXES)


def docstring_summary(obj):
    """First line of the docstring, without a trailing period."""
    doc = inspect.getdoc(obj) or ""
    return doc.strip().split("\n")[0].strip().rstrip(".")


def first_paragraph(doc):
    """First paragraph of a docstring (everything before the first blank line)."""
    return (doc or "").strip().split("\n\n")[0]


def format_signature(name, obj):
    """Render ``name(params)``, or just ``name`` when no signature exists."""
    try:
        return f"{name}{inspect.signature(obj)}"
    except (ValueError, TypeError):
        return name


def iter_public_members(cls, include_inherited):
    """Yield (name, member) for the class's public methods and properties.

    Honors the quartodoc ``include_inherited`` flag: with it, members from
    base classes are included (e.g. ``Table``); without it, only members
    defined on the class itself.
    """
    names = dir(cls) if include_inherited else vars(cls)
    for name in sorted(names):
        if name.startswith("_"):
            continue
        member = inspect.getattr_static(cls, name, None)
        if isinstance(member, (staticmethod, classmethod)):
            member = member.__func__
        if callable(member) or isinstance(member, property):
            yield (name, member)


# =============================================================================
# llms.txt (compact index)
# =============================================================================


def render_api_index(config):
    lines = ["### API Reference", ""]
    for title, desc, entries in iter_api_sections(config):
        lines.append(f"#### {title}")
        if desc:
            lines.append(f"> {desc}")
        lines.append("")
        for entry in entries:
            if entry[0] == "page":
                (_, path, summary_name, summary_desc, _) = entry
                url = f"{SITE_URL}reference/{path}.html"
                lines.append(f"- [{summary_name}]({url}): {summary_desc}")
            else:
                (_, name, package, _) = entry
                summary = docstring_summary(resolve_item(package, name))
                url = f"{SITE_URL}reference/{name}.html"
                suffix = f": {summary}" if summary else ""
                lines.append(f"- [{name}]({url}){suffix}")
        lines.append("")
    return lines


def iter_cli_commands():
    """Yield (key, cmd, invocation) in the curated group order."""
    main_commands = cli_ref.load_main_commands()
    (catalog_commands, _) = cli_ref.load_catalog_commands()
    for _, names in cli_ref.MAIN_GROUPS:
        for name in names:
            (cmd, invocation) = main_commands[name]
            yield (name, cmd, invocation)
    for _, names in cli_ref.CATALOG_GROUPS:
        for name in names:
            (cmd, invocation) = catalog_commands[name]
            yield (f"catalog/{name}", cmd, invocation)


def cli_page_url(key):
    page = cli_ref.filename_for(key.rpartition("/")[2]).removesuffix(".qmd")
    prefix = "catalog/" if key.startswith("catalog/") else ""
    return f"{SITE_URL}api_reference/cli/{prefix}{page}.html"


def render_cli_index():
    lines = ["### CLI Reference", ""]
    for key, cmd, invocation in iter_cli_commands():
        short_help = cmd.get_short_help_str(limit=120)
        lines.append(f"- [{invocation}]({cli_page_url(key)}): {short_help}")
    lines.append("")
    return lines


def split_frontmatter(text):
    """Return (frontmatter dict, body) for a qmd file."""
    match = re.match(r"\A---\n(.*?)\n---\n(.*)\Z", text, re.DOTALL)
    if not match:
        return ({}, text)
    return (yaml.safe_load(match.group(1)) or {}, match.group(2))


def page_description(frontmatter, body):
    """One-line description: frontmatter wins, else first prose sentence."""
    if frontmatter.get("description"):
        return str(frontmatter["description"]).strip()
    for paragraph in body.split("\n\n"):
        stripped = paragraph.strip()
        if not stripped or stripped.startswith(
            ("```", ":::", "#", "|", "-", "!", "<", "{")
        ):
            continue
        text = re.sub(r"\[([^]]*)\]\([^)]*\)", r"\1", stripped)
        text = " ".join(text.split())
        match = re.match(r"(.+?\.)(?:\s|$)", text)
        return (match.group(1) if match else text).rstrip(".")
    return ""


def load_guide_page(href):
    """Return (title, description, body) for a sidebar guide page."""
    text = (DOCS_DIR / href).read_text()
    (frontmatter, body) = split_frontmatter(text)
    title = frontmatter.get("title", Path(href).stem.replace("_", " "))
    return (title, page_description(frontmatter, body), body)


def render_guides_index(config):
    lines = ["### Guides", ""]
    current_group = None
    for group_title, _, href, link_text in iter_sidebar_pages(config):
        if group_title != current_group:
            if current_group is not None:
                lines.append("")
            lines.append(f"#### {group_title}")
            lines.append("")
            current_group = group_title
        (title, description, _) = load_guide_page(href)
        url = SITE_URL + href.removesuffix(".qmd") + ".html"
        suffix = f": {description}" if description else ""
        lines.append(f"- [{link_text or title}]({url}){suffix}")
    lines.append("")
    return lines


def render_llms_txt(config, package_name, package_description):
    lines = [f"# {package_name}", ""]
    if package_description:
        lines.extend((f"> {package_description}", ""))
    lines.extend(("## Docs", ""))
    lines.extend(render_api_index(config))
    lines.extend(render_cli_index())
    lines.extend(render_guides_index(config))
    return "\n".join(lines).rstrip() + "\n"


# =============================================================================
# llms-full.txt (comprehensive)
# =============================================================================


def render_member_details(cls_name, member_name, member, vendored):
    """Render one method or property of a class."""
    if isinstance(member, property):
        doc = first_paragraph(inspect.getdoc(member.fget) if member.fget else "")
        header = f"{cls_name}.{member_name} (property)"
    else:
        doc = inspect.getdoc(member) or ""
        if vendored:
            doc = first_paragraph(doc)
        header = format_signature(f"{cls_name}.{member_name}", member)
    if doc:
        return f"{header}\n\n{doc}"
    return header


def render_module_details(name, module):
    """Render a documented module: docstring plus public member one-liners."""
    lines = [f"### {name} (module)", ""]
    doc = inspect.getdoc(module)
    if doc:
        lines.extend((doc, ""))
    for member_name in sorted(vars(module)):
        if member_name.startswith("_"):
            continue
        member = getattr(module, member_name)
        if inspect.ismodule(member):
            continue
        summary = docstring_summary(member)
        suffix = f": {summary}" if summary else ""
        lines.append(f"- {member_name}{suffix}")
    return "\n".join(lines).rstrip()


def render_class_details(name, cls, vendored, include_inherited):
    """Render a class: signature, docstring, then each public member."""
    doc = inspect.getdoc(cls) or ""
    if vendored:
        doc = first_paragraph(doc)
    parts = [f"### {format_signature(name, cls)}"]
    if doc:
        parts.append(doc)
    parts.extend(
        render_member_details(name, member_name, member, vendored)
        for (member_name, member) in iter_public_members(cls, include_inherited)
    )
    return "\n\n".join(parts)


def render_api_details(name, package, include_inherited):
    """Render full documentation for one quartodoc item."""
    obj = resolve_item(package, name)
    vendored = is_vendored(package)
    if inspect.ismodule(obj):
        return render_module_details(name, obj)
    if inspect.isclass(obj):
        return render_class_details(name, obj, vendored, include_inherited)
    doc = inspect.getdoc(obj) or ""
    if vendored:
        doc = first_paragraph(doc)
    header = f"### {format_signature(name, obj)}"
    return f"{header}\n\n{doc}" if doc else header


def render_api_full(config, package_name):
    lines = [
        SEP_LINE,
        f"This is the API documentation for the {package_name} library.",
        SEP_LINE,
        "",
    ]
    for title, desc, entries in iter_api_sections(config):
        lines.append(f"## {title}")
        if desc:
            lines.extend(("", desc))
        lines.append("")
        for entry in entries:
            if entry[0] == "page":
                (_, _, summary_name, summary_desc, subentries) = entry
                lines.append(f"### {summary_name}")
                if summary_desc:
                    lines.extend(("", summary_desc))
                lines.append("")
                items = subentries
            else:
                items = (entry,)
            for _, name, package, include_inherited in items:
                details = render_api_details(name, package, include_inherited)
                # Demote page-grouped items below their page heading.
                if entry[0] == "page":
                    details = details.replace("### ", "#### ", 1)
                lines.extend((details, ""))
    return lines


def render_cli_full():
    lines = [
        SEP_LINE,
        "This is the CLI documentation for the xorq command-line tool.",
        SEP_LINE,
        "",
    ]
    for _, cmd, invocation in iter_cli_commands():
        ctx = click.Context(cmd, info_name=invocation)
        lines.extend((f"## {invocation}", "", "```", cmd.get_help(ctx), "```", ""))
    return lines


def render_guides_full(config):
    lines = [
        SEP_LINE,
        "This is the user guide documentation for xorq.",
        SEP_LINE,
        "",
    ]
    for group_title, section_title, href, link_text in iter_sidebar_pages(config):
        (title, _, body) = load_guide_page(href)
        breadcrumb = " — ".join(
            part for part in (group_title, section_title, link_text or title) if part
        )
        lines.extend((f"## {breadcrumb}", "", body.strip(), ""))
    return lines


def render_llms_full_txt(config, package_name, package_description):
    lines = [f"# {package_name}", ""]
    if package_description:
        lines.extend((f"> {package_description}", ""))
    lines.extend(render_api_full(config, package_name))
    lines.extend(render_cli_full())
    lines.extend(render_guides_full(config))
    return "\n".join(lines).rstrip() + "\n"


# =============================================================================
# Output
# =============================================================================


def main():
    config = read_quarto_config()
    (package_name, package_description) = read_package_metadata()

    llms_txt = DOCS_DIR / "llms.txt"
    llms_txt.write_text(render_llms_txt(config, package_name, package_description))
    print(
        f"generated {llms_txt.relative_to(DOCS_DIR)} ({llms_txt.stat().st_size:,} bytes)"
    )

    llms_full = DOCS_DIR / "llms-full.txt"
    llms_full.write_text(
        render_llms_full_txt(config, package_name, package_description)
    )
    print(
        f"generated {llms_full.relative_to(DOCS_DIR)} ({llms_full.stat().st_size:,} bytes)"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
