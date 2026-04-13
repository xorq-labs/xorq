#!/usr/bin/env python
"""Seed a catalog with PlotNode entries for testing the TUI image display.

Creates a mix of regular source/expr entries and PlotNode entries that
produce PNG images, so you can verify the TUI renders plots via the
Kitty graphics protocol instead of a DataTable.

Usage:
    python scripts/catalog_plot_seed.py [/path/to/catalog]

Defaults to /tmp/plot-seed-catalog.
"""

import io
import sys
from pathlib import Path

import xorq.api as xo
from xorq.catalog.catalog import Catalog
from xorq.common.utils.plot_utils import make_plot_expr
from xorq.vendor.ibis.expr.datatypes import binary


def _scatter_plot(df):
    """Create a scatter plot of bill length vs body mass and return PNG bytes."""
    import matplotlib as mpl  # noqa: PLC0415

    mpl.use("Agg")
    import matplotlib.pyplot as plt  # noqa: PLC0415

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(df["bill_length_mm"], df["body_mass_g"], alpha=0.6, s=20)
    ax.set_xlabel("Bill Length (mm)")
    ax.set_ylabel("Body Mass (g)")
    ax.set_title("Penguin Bill Length vs Body Mass")
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=100)
    plt.close(fig)
    return buf.getvalue()


def _histogram_plot(df):
    """Create a histogram of flipper lengths and return PNG bytes."""
    import matplotlib as mpl  # noqa: PLC0415

    mpl.use("Agg")
    import matplotlib.pyplot as plt  # noqa: PLC0415

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(df["flipper_length_mm"].dropna(), bins=20, edgecolor="black", alpha=0.7)
    ax.set_xlabel("Flipper Length (mm)")
    ax.set_ylabel("Count")
    ax.set_title("Distribution of Flipper Lengths")
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=100)
    plt.close(fig)
    return buf.getvalue()


def _bar_chart_plot(df):
    """Create a bar chart of mean body mass by species and return PNG bytes."""
    import matplotlib as mpl  # noqa: PLC0415

    mpl.use("Agg")
    import matplotlib.pyplot as plt  # noqa: PLC0415

    grouped = df.groupby("species")["body_mass_g"].mean().sort_values()
    fig, ax = plt.subplots(figsize=(6, 4))
    grouped.plot(kind="barh", ax=ax, color=["#2BBE75", "#4AA8EC", "#F5CA2C"])
    ax.set_xlabel("Mean Body Mass (g)")
    ax.set_title("Body Mass by Species")
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=100)
    plt.close(fig)
    return buf.getvalue()


def seed(catalog):
    # --- Source table: penguins ---
    penguins = xo.memtable(
        {
            "species": ["Adelie"] * 10 + ["Chinstrap"] * 10 + ["Gentoo"] * 10,
            "island": ["Torgersen"] * 10 + ["Dream"] * 10 + ["Biscoe"] * 10,
            "bill_length_mm": [
                39.1,
                39.5,
                40.3,
                36.7,
                39.3,
                38.9,
                42.0,
                41.1,
                38.6,
                34.6,
                46.5,
                50.0,
                51.3,
                45.4,
                52.7,
                49.6,
                46.9,
                48.7,
                50.2,
                45.1,
                46.1,
                50.0,
                48.7,
                50.0,
                47.6,
                46.5,
                45.4,
                46.7,
                43.3,
                46.8,
            ],
            "bill_depth_mm": [
                18.7,
                17.4,
                18.0,
                19.3,
                20.6,
                17.8,
                19.6,
                18.2,
                21.2,
                21.1,
                17.9,
                19.5,
                18.2,
                18.7,
                19.0,
                18.2,
                16.6,
                18.3,
                18.7,
                17.0,
                13.2,
                16.3,
                14.1,
                15.2,
                14.5,
                13.5,
                13.7,
                15.3,
                13.4,
                15.4,
            ],
            "flipper_length_mm": [
                181,
                186,
                195,
                193,
                190,
                181,
                190,
                198,
                185,
                195,
                192,
                196,
                198,
                188,
                197,
                193,
                192,
                195,
                198,
                186,
                215,
                218,
                215,
                220,
                215,
                210,
                211,
                216,
                209,
                215,
            ],
            "body_mass_g": [
                3750,
                3800,
                3250,
                3450,
                3650,
                3625,
                4250,
                3200,
                3800,
                3700,
                3500,
                3900,
                4100,
                3525,
                3725,
                3950,
                3875,
                3700,
                3800,
                3550,
                4500,
                5700,
                4450,
                5200,
                4750,
                4550,
                4400,
                5050,
                4100,
                4850,
            ],
        }
    )
    catalog.add(penguins, aliases=("penguins",))

    # --- Regular expr entries ---
    adelie = penguins.filter(penguins.species == "Adelie")
    catalog.add(adelie, aliases=("adelie-penguins",))

    species_stats = penguins.group_by("species").agg(
        mean_mass=penguins.body_mass_g.mean(),
        mean_flipper=penguins.flipper_length_mm.mean(),
        count=penguins.species.count(),
    )
    catalog.add(species_stats, aliases=("species-stats",))

    # --- PlotNode entries ---
    scatter_expr = make_plot_expr(penguins, _scatter_plot, binary, name="scatter_plot")
    catalog.add(scatter_expr, aliases=("scatter-bill-vs-mass",))

    histogram_expr = make_plot_expr(
        penguins, _histogram_plot, binary, name="flipper_histogram"
    )
    catalog.add(histogram_expr, aliases=("flipper-length-histogram",))

    bar_expr = make_plot_expr(
        penguins, _bar_chart_plot, binary, name="species_bar_chart"
    )
    catalog.add(bar_expr, aliases=("species-mass-barchart",))

    return catalog


if __name__ == "__main__":
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("/tmp/plot-seed-catalog")
    catalog = Catalog.from_repo_path(path, init=True)
    seed(catalog)
    print(f"Catalog seeded at {path} with {len(list(catalog.list()))} entries")
