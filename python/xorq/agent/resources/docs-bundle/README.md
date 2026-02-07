# xorq Documentation Bundle

This directory contains a minified, self-contained documentation bundle for xorq, inspired by Vercel's Next.js AGENTS.md approach.

**Package Location:** `python/xorq/agent/resources/docs-bundle/` (ships with the CLI)

## What's Inside

1. **DOCS_INDEX.txt** - Compressed pipe-delimited index (3KB) in Vercel style
2. **xorq-docs.tar.gz** - Complete documentation archive (68KB compressed, 231KB uncompressed)
3. **metadata.json** - Bundle metadata (106 files tracked)
4. **AGENTS_MD_SNIPPET.txt** - Ready-to-use snippet for AGENTS.md

## How It Works

When users run `xorq agents init --agents claude`, the CLI:

1. **Injects minified docs index** into AGENTS.md/CLAUDE.md
2. **References docs directory** (`.xorq-docs/` or `docs/`)
3. **Enables retrieval-led reasoning** with the instruction:
   > "IMPORTANT: Prefer retrieval-led reasoning over pre-training-led reasoning for xorq tasks"

## Compressed Index Format

The index uses pipe-delimited format for maximum compression:

```
[xorq Docs Index]|root: .xorq-docs
|IMPORTANT: Prefer retrieval-led reasoning over pre-training-led reasoning for xorq tasks
|IMPORTANT: All xorq expressions must be deferred - no eager pandas/NumPy operations
|api_reference/backends:{env_variables.qmd,index.qmd,overview.qmd,profiles_api.qmd,supported_backends.qmd}
|api_reference/cli:{build.qmd,index.qmd,init.qmd,lineage.qmd,ps.qmd,run.qmd,...}
|concepts/core_concepts:{batch_processing_llms.qmd,build_system.qmd,...}
...
```

Each line maps a directory path to the doc files it contains, using `{}` braces for file lists.

## Size Comparison

| Component | Size | Compression |
|-----------|------|-------------|
| Compressed Index | 3 KB | 80% reduction from full paths |
| Tarball | 68 KB | 70% reduction from raw files |
| Full AGENTS.md | 11 KB | Includes index + workflow instructions |

## Vercel Approach Advantages

Based on Vercel's blog post findings:

1. **No decision point** - Docs are always available in context (passive vs active retrieval)
2. **Consistent availability** - Index is in system prompt for every turn
3. **No ordering issues** - No sequencing decisions ("read docs first" vs "explore project first")
4. **100% pass rate** - In Vercel's evals, passive context (AGENTS.md) achieved 100% vs skills at 79%

## Building the Bundle

To rebuild the bundle (e.g., when docs are updated):

```bash
python3 build_docs_bundle.py
```

This regenerates:
- `DOCS_INDEX.txt` - Compressed index
- `xorq-docs.tar.gz` - Documentation archive
- `metadata.json` - Bundle metadata
- `AGENTS_MD_SNIPPET.txt` - AGENTS.md snippet

## CLI Integration

The bundle is automatically used by:

```python
# In python/xorq/agent/onboarding.py
def _get_docs_index() -> str:
    """Get the minified documentation index in Vercel blog style."""
    import xorq
    xorq_package_dir = Path(xorq.__file__).parent
    # Docs bundle is shipped with the package
    docs_bundle = xorq_package_dir / "agent" / "resources" / "docs-bundle"
    index_file = docs_bundle / "DOCS_INDEX.txt"

    if index_file.exists():
        return index_file.read_text()

    # Fallback to minimal inline index
    return "..."
```

The index is then injected into AGENTS.md when users run:
```bash
xorq agents init --agents claude
```

## Documentation Coverage

The bundle includes **106 documentation files** across:

- **Getting Started** (8 files) - Installation, quickstart, first expression
- **Core Concepts** (18 files) - Deferred execution, caching, multi-engine, UDXFs
- **Tutorials** (20 files) - Core, ML, AI, analytics tutorials
- **Guides** (23 files) - Data, ML, performance, platform workflows
- **API Reference** (17 files) - CLI commands, backends, profiles
- **Troubleshooting** (7 files) - Common errors, installation, performance

## Future Enhancements

1. **Auto-extraction** - CLI could extract tarball to `.xorq-docs/` on first use
2. **Version matching** - Bundle docs matching installed xorq version
3. **Incremental updates** - Only extract changed files
4. **Package distribution** - Include bundle in pip/conda packages

## References

- [Vercel Blog: AGENTS.md outperforms skills](https://vercel.com/blog/agents-md-outperforms-skills)
- Compressed index: 80% reduction (similar to Vercel's 8KB from 40KB)
- Passive context > Active retrieval for framework knowledge

---

**Generated:** 2026-02-07
**xorq version:** Pre-1.0
**Total files:** 106
**Compressed size:** 71 KB (index + tarball + metadata)
