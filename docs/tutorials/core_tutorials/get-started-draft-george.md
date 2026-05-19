# Get started with `xorq init` (terse)

Scaffold a xorq project, load Moneyball CSVs, build a top-batters leaderboard. Pure xorq, no ibis import, no shell tricks.

## Prerequisites

[uv](https://docs.astral.sh/uv/):

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

## Scaffold

xorq requires Python 3.13. Make sure uv has it, then scaffold and sync against it explicitly:

```bash
uv python install 3.13
uvx -p3.13 xorq@latest init --path moneyball
cd moneyball
uv sync -p3.13
source .venv/bin/activate
```

`uvx` bootstraps xorq once; from here on the `xorq` on your PATH comes from the project's pinned `.venv`.

## Get the data

```bash
curl -fL -o Batting.csv https://raw.githubusercontent.com/xorq-labs/baseballdatabank/master/core/Batting.csv
curl -fL -o People.csv https://raw.githubusercontent.com/xorq-labs/baseballdatabank/master/core/People.csv
```

## Write the expression

Replace `expr.py` with:

```python
# expr.py
import xorq.api as xo

# in-process backend; no server, no config
con = xo.connect()

# register CSVs as typed expressions, no read until build time
batting = xo.deferred_read_csv(con=con, path="Batting.csv", table_name="Batting")
people  = xo.deferred_read_csv(con=con, path="People.csv",  table_name="People")

# attach player bio columns to each player-season row
batting = batting.join(
    people["playerID", "nameFirst", "nameLast", "bats", "throws", "birthYear"],
    "playerID",
)

# HBP and SF are null in older seasons, coalesce so arithmetic doesn't null-poison
batting = batting.mutate(
    HBP=batting.HBP.fillna(0),
    SF=batting.SF.fillna(0),
)

# on-base percentage: (H + BB + HBP) / (AB + BB + HBP + SF)
batting = batting.mutate(
    OBP=(batting.H + batting.BB + batting.HBP)
        / (batting.AB.cast("float64") + batting.BB + batting.HBP + batting.SF)
)

# modern era, AL/NL only, real hitters (>=100 AB so batting avg is meaningful)
batting = batting.filter(
    batting.lgID.isin(["AL", "NL"]),
    batting.yearID > 1965,
    batting.AB >= 100,
)

# batting average per player-season
ranked = batting.mutate(batting_avg=batting.H / batting.AB.cast("float64"))

# rank within each (league, year) by batting_avg, descending
win    = xo.window(group_by=["lgID", "yearID"], order_by=xo.desc("batting_avg"))
ranked = ranked.mutate(rank=xo.row_number().over(win) + 1)

# top 10 per league-year; `expr` is what `xorq build` compiles
expr = ranked.filter(ranked.rank <= 10).drop("rank")
```

Nothing has run yet. `expr` is a description of the computation; `xorq build` compiles it.

## Build and run

```bash
xorq build expr.py -e expr --builds-dir builds
```

Output lands in `builds/$BUILD_HASH/`. The hash is derived from the expression graph and your lock file. Grab the most recent build dir into a shell var so you don't have to think about it:

```bash
# grab build hash
BUILD_HASH=$(echo builds/*/expr.yaml | cut -d/ -f2)
xorq run "builds/$BUILD_HASH" \
    --output-path top_batters.parquet
```

Each build is self-contained:

```bash
ls builds/$BUILD_HASH
```

`expr.yaml` is the serialized graph; `requirements.txt` is pinned from your lock. Hand the directory to anyone with `xorq` and `xorq run` reproduces the result.

If `expr.py` changes its build changes. Edit `expr.py` and run `xorq build` again — bump the year filter from 1965 to 1970:

```bash
sed -i '' 's/1965/1970/' expr.py
xorq build expr.py -e expr --builds-dir builds
```

You'll get a new directory next to the old one:

```bash
tree -L 1 builds
```

Nothing is overwritten. Re-running `xorq build` with no changes returns the same hash, no new directory.

## Inspect

```bash
xorq run "builds/$BUILD_HASH" \
    --format csv \
    --limit 5 \
    -o /dev/stdout
```

## What you used

- `xo.deferred_read_csv` — lazy CSV registration, schema inferred at build time
- `.join`, `.mutate`, `.filter` — expression construction, nothing executes
- `xo.window` / `xo.row_number` / `xo.desc` — windowing, all on the `xo` namespace
- `xorq build` / `xorq run` — compile to `builds/$BUILD_HASH/`, then execute

## Next

- [Why deferred execution?](/concepts/understanding_xorq/why_deferred_execution.qmd)
- [Working with the catalog](/tutorials/core_tutorials/working_with_the_catalog.qmd)
- [`xorq build` reference](/api_reference/cli/build.qmd)
