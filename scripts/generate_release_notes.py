#!/usr/bin/env python3
"""Convert CHANGELOG.md to release_notes_generated.qmd format for documentation."""

import re
from pathlib import Path


# Section mapping from CHANGELOG to release notes format
SECTION_MAP = {
    "Added": "Features",
    "Changed": "Refactors",
    "Fixed": "Bug Fixes",
    "Removed": "Deprecations",
    "Security": "Security",
}


def parse_changelog(changelog_path: Path) -> list[dict]:
    """Parse CHANGELOG.md and extract release information."""
    content = changelog_path.read_text()

    releases = []
    current_release = None
    current_section = None

    for line in content.split("\n"):
        # Match version header: ## [0.3.7] - 2026-01-08
        version_match = re.match(r"^## \[([^\]]+)\]\s*-\s*(.+)$", line)
        if version_match:
            if current_release:
                releases.append(current_release)
            current_release = {
                "version": version_match.group(1),
                "date": version_match.group(2),
                "sections": {},
            }
            current_section = None
            continue

        # Skip "### Details" line
        if line.strip() == "### Details":
            continue

        # Match section header: #### Added
        section_match = re.match(r"^####\s+(.+)$", line)
        if section_match:
            section_name = section_match.group(1).strip()
            current_section = SECTION_MAP.get(section_name, section_name)
            if current_release and current_section:
                current_release["sections"][current_section] = []
            continue

        # Match changelog item: - Item text by @user in [#123](url)
        if line.strip().startswith("-") and current_release and current_section:
            # Extract PR number and URL if present
            pr_match = re.search(r"\[#(\d+)\]\((https://[^)]+)\)", line)
            pr_number = pr_match.group(1) if pr_match else None
            pr_url = pr_match.group(2) if pr_match else None

            # Extract username if present
            username_match = re.search(r"by @(\w+)", line)
            username = username_match.group(1) if username_match else None

            # Clean up the item text (remove markdown links, keep text)
            item_text = line.strip()[1:].strip()  # Remove leading "-"
            # Remove "by @user" and "in [#123](url)" parts for cleaner text
            item_text = re.sub(
                r"\s+by @[\w\[\]]+", "", item_text
            )  # Handle [bot] in username
            item_text = re.sub(r"\s+in \[#\d+\]\([^)]+\)", "", item_text)

            # Ensure sentence case: capitalize first letter if not already capitalized
            if item_text and item_text[0].islower():
                item_text = item_text[0].upper() + item_text[1:]

            # Format: Item text (#123) by @user
            formatted_item = item_text
            if pr_number:
                pr_link = f"[#{pr_number}]({pr_url})" if pr_url else f"#{pr_number}"
                formatted_item = f"{item_text} ({pr_link})"
            if username:
                formatted_item = f"{formatted_item} by @{username}"

            current_release["sections"][current_section].append(formatted_item)

    # Add the last release
    if current_release:
        releases.append(current_release)

    return releases


def format_release_notes(releases: list[dict]) -> str:
    """Format releases into Quarto Markdown format."""
    lines = []

    for release in releases:
        version = release["version"]
        date = release["date"]

        # Version header with date in format: Version (YYYY-MM-DD)
        lines.append(f"## {version} ({date})")
        lines.append("")

        # Process sections in order
        section_order = [
            "Features",
            "Bug Fixes",
            "Refactors",
            "Deprecations",
            "Security",
            "Documentation",
            "Performance",
        ]

        for section_name in section_order:
            if (
                section_name in release["sections"]
                and release["sections"][section_name]
            ):
                items = release["sections"][section_name]
                lines.append(f"### {section_name}")
                lines.append("")
                for item in items:
                    lines.append(f"- {item}")
                lines.append("")

        # Add any remaining sections not in the order
        for section_name, items in release["sections"].items():
            if section_name not in section_order and items:
                lines.append(f"### {section_name}")
                lines.append("")
                for item in items:
                    lines.append(f"- {item}")
                lines.append("")

    return "\n".join(lines)


def main():
    """Main function to generate release notes."""
    repo_root = Path(__file__).parent.parent
    changelog_path = repo_root / "CHANGELOG.md"
    generated_path = repo_root / "docs" / "release_notes_generated.qmd"
    release_notes_path = repo_root / "docs" / "release_notes.qmd"

    if not changelog_path.exists():
        print(f"Error: {changelog_path} not found")
        return 1

    releases = parse_changelog(changelog_path)
    generated_content = format_release_notes(releases)

    # Write the generated content
    generated_path.write_text(generated_content)

    # Create the full release_notes.qmd with the generated content embedded
    release_notes_content = f"""---
title: 'Release Notes'
---

```{{=html}}
<!-- This file is automatically generated from CHANGELOG.md -->
<style>
/* Release Notes Styling */
.release-notes-intro {{
  font-size: 1.125rem;
  line-height: 1.7;
  color: inherit;
  margin-bottom: 3rem;
  max-width: 65ch;
}}

.release-version {{
  position: relative;
  margin-top: 3rem;
  margin-bottom: 2rem;
  padding-bottom: 1rem;
  border-bottom: 2px solid rgba(121, 226, 255, 0.2);
}}

[data-bs-theme="light"] .release-version {{
  border-bottom-color: rgba(5, 24, 26, 0.15);
}}

.release-version:first-of-type {{
  margin-top: 0;
}}

.release-version h2 {{
  display: inline-flex;
  align-items: center;
  gap: 0.75rem;
  font-size: 1.75rem;
  font-weight: 500;
  margin-bottom: 0.5rem;
  color: #79E2FF;
}}

[data-bs-theme="light"] .release-version h2 {{
  color: #05181A;
}}

.release-version .version-badge {{
  display: inline-block;
  padding: 0.25rem 0.75rem;
  background: rgba(121, 226, 255, 0.15);
  border: 1px solid rgba(121, 226, 255, 0.3);
  border-radius: 0.5rem;
  font-size: 0.875rem;
  font-weight: 500;
  color: #79E2FF;
  font-family: 'Space Mono', monospace;
}}

[data-bs-theme="light"] .release-version .version-badge {{
  background: rgba(5, 24, 26, 0.08);
  border-color: rgba(5, 24, 26, 0.2);
  color: #05181A;
}}

.release-version .version-date {{
  font-size: 0.875rem;
  color: rgba(255, 255, 255, 0.5);
  font-weight: 400;
  margin-left: 0.5rem;
}}

[data-bs-theme="light"] .release-version .version-date {{
  color: rgba(5, 24, 26, 0.6);
}}

.release-section {{
  margin-top: 2rem;
  margin-bottom: 1.5rem;
}}

.release-section h3 {{
  font-size: 0.75rem;
  font-weight: 600;
  text-transform: uppercase;
  letter-spacing: 0.1em;
  color: rgba(121, 226, 255, 0.9);
  margin-bottom: 1rem;
  padding: 0.5rem 0;
  border-bottom: 1px solid rgba(121, 226, 255, 0.15);
  display: inline-block;
  width: 100%;
}}

[data-bs-theme="light"] .release-section h3 {{
  color: rgba(5, 24, 26, 0.8);
  border-bottom-color: rgba(5, 24, 26, 0.15);
}}

.release-section ul {{
  list-style: none;
  padding-left: 0;
  margin: 0;
}}

.release-section li {{
  position: relative;
  padding-left: 1.5rem;
  margin-bottom: 0.75rem;
  line-height: 1.6;
  color: rgba(255, 255, 255, 0.85);
}}

[data-bs-theme="light"] .release-section li {{
  color: rgba(5, 24, 26, 0.85);
}}

.release-section li::before {{
  content: "→";
  position: absolute;
  left: 0;
  color: #79E2FF;
  font-weight: 400;
}}

[data-bs-theme="light"] .release-section li::before {{
  color: #05181A;
}}

.release-section li a {{
  color: #79E2FF;
  text-decoration: none;
  font-weight: 500;
  transition: opacity 0.2s ease;
}}

[data-bs-theme="light"] .release-section li a {{
  color: #05181A;
}}

.release-section li a:hover {{
  opacity: 0.8;
  text-decoration: underline;
}}

.release-section li code {{
  background: rgba(121, 226, 255, 0.1);
  padding: 0.125rem 0.375rem;
  border-radius: 0.25rem;
  font-size: 0.875em;
  color: #79E2FF;
  font-family: 'Space Mono', monospace;
}}

[data-bs-theme="light"] .release-section li code {{
  background: rgba(5, 24, 26, 0.1);
  color: #05181A;
}}

.release-footer {{
  margin-top: 4rem;
  padding-top: 2rem;
  border-top: 1px solid rgba(121, 226, 255, 0.15);
  font-size: 0.875rem;
  color: rgba(255, 255, 255, 0.6);
}}

[data-bs-theme="light"] .release-footer {{
  border-top-color: rgba(5, 24, 26, 0.15);
  color: rgba(5, 24, 26, 0.6);
}}

.release-footer a {{
  color: #79E2FF;
  text-decoration: none;
}}

[data-bs-theme="light"] .release-footer a {{
  color: #05181A;
}}

.release-footer a:hover {{
  text-decoration: underline;
}}
</style>
```

<div class="release-notes-intro">
Stay up to date with the latest changes, improvements, and new features in Xorq. Each release includes bug fixes, new capabilities, and performance improvements to help you build reproducible ML infrastructure.
</div>

{generated_content}

<div class="release-footer">
The release notes are automatically generated from the <a href="https://github.com/xorq-labs/xorq/blob/main/CHANGELOG.md">CHANGELOG.md</a> file.
</div>
"""

    release_notes_path.write_text(release_notes_content)
    print(f"Generated {generated_path} and updated {release_notes_path}")

    return 0


if __name__ == "__main__":
    exit(main())
