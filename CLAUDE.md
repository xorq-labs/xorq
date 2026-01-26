## xorq Expression Framework

This project uses **xorq** for deferred data pipelines.
Run `xorq agents prime` for dynamic workflow context, or install hooks (`xorq agents hooks install`) for auto-injection.

## Mandatory Workflow
1. `xorq agents vignette list` - discover patterns
2. `xorq agents vignette scaffold <name> --dest reference.py` - get template
3. Follow the scaffolded pattern exactly
4. `xorq build <file>.py -e expr` - build expression
5. `xorq catalog add .xorq/builds/<hash> --alias <name>` - register

**Quick reference:**
- `xorq catalog ls` - Find available expressions
- `xorq build expr.py -e expr` - Build expression
- `xorq catalog add builds/<hash> --alias name` - Catalog build
- `xorq run <alias> -f arrow -o /dev/stdout | ...` - Stream expression
- `xorq agents prime` - Get current workflow context and state

**Key principle:** Everything is a deferred expression - no eager pandas/NumPy!
For ML patterns, use `xorq agents vignette` for deferred sklearn guidance.

For full workflow details: `xorq agents prime`
