"""Demonstrates the Catalog with an S3-backed content store (pointer backend).

With the pointer backend, catalog entries are stored as lightweight pointer files
in git while the actual data archives live in an S3 bucket. This keeps git
repos small and enables sharing large artifacts via object storage.

Requirements:
    - boto3 installed (``pip install boto3``)
    - S3 credentials available (env vars, AWS profile, or explicit)

Configure via environment variables::

    export XORQ_CONTENT_STORE_S3_BUCKET=my-bucket
    export XORQ_CONTENT_STORE_S3_CATALOG_ID=my-catalog
    export XORQ_CONTENT_STORE_S3_PREFIX=catalogs/dev/       # optional
    export XORQ_CONTENT_STORE_S3_REGION=us-east-1            # optional
    export XORQ_CONTENT_STORE_S3_AWS_ACCESS_KEY_ID=...       # optional if using AWS profile
    export XORQ_CONTENT_STORE_S3_AWS_SECRET_ACCESS_KEY=...   # optional if using AWS profile

Or for S3-compatible services (R2, MinIO, etc.)::

    export XORQ_CONTENT_STORE_S3_HOST=your-account.r2.cloudflarestorage.com
    export XORQ_CONTENT_STORE_S3_PROTOCOL=https
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import click

import xorq.api as xo
from xorq.catalog.catalog import Catalog
from xorq.catalog.constants import POINTER_SUFFIX
from xorq.catalog.content_store import S3ContentStoreConfig


def main() -> None:
    # -------------------------------------------------------------------
    # 1. Build the S3 content store config
    # -------------------------------------------------------------------

    # Option A: load everything from XORQ_CONTENT_STORE_S3_* env vars
    config = S3ContentStoreConfig.from_env()

    _tmpdirs: list[tempfile.TemporaryDirectory] = []

    # Option B: explicit construction (uncomment to use instead)
    # config = S3ContentStoreConfig(
    #     catalog_id="my-catalog",
    #     bucket="my-bucket",
    #     prefix="catalogs/dev/",
    #     region="us-east-1",
    # )

    click.echo(f"Catalog ID : {config.catalog_id}")
    click.echo(f"Bucket     : {config.bucket}")
    click.echo(f"Prefix     : {config.prefix or '(none)'}")

    # -------------------------------------------------------------------
    # 2. Verify bucket connectivity
    # -------------------------------------------------------------------

    check = config.check_bucket(check_write=True)
    click.echo(f"\nBucket check: {check}")

    # -------------------------------------------------------------------
    # 3. Initialize a catalog with the pointer backend
    # -------------------------------------------------------------------

    _td1 = tempfile.TemporaryDirectory()
    _tmpdirs.append(_td1)
    catalog_dir = Path(_td1.name) / "s3-catalog"
    catalog = Catalog.from_repo_path(
        catalog_dir,
        init=True,
        content_store_config=config,
    )
    click.echo(f"\nCatalog directory: {catalog_dir}")
    click.echo(f"Backend type     : {type(catalog.backend).__name__}")

    # -------------------------------------------------------------------
    # 4. Create expressions and add them to the catalog
    # -------------------------------------------------------------------

    flights = xo.memtable(
        {
            "origin": ["JFK", "LAX", "ORD", "JFK", "LAX"],
            "carrier": ["AA", "UA", "AA", "UA", "AA"],
            "dep_delay": [10.0, -5.0, 30.0, 15.0, -2.0],
            "distance": [2475, 1745, 740, 1300, 2475],
        },
        name="flights",
    )

    delays_by_origin = flights.group_by("origin").agg(
        flight_count=flights.origin.count(),
        avg_delay=flights.dep_delay.mean(),
    )

    catalog.add(delays_by_origin, aliases=("delays-by-origin",), sync=False)
    click.echo("\nAdded delays-by-origin")

    delays_by_carrier = flights.group_by("carrier").agg(
        flight_count=flights.carrier.count(),
        avg_delay=flights.dep_delay.mean(),
        total_distance=flights.distance.sum(),
    )

    catalog.add(delays_by_carrier, aliases=("delays-by-carrier",), sync=False)
    click.echo("Added delays-by-carrier")

    click.echo(f"\nCatalog entries : {catalog.list()}")
    click.echo(f"Catalog aliases : {catalog.list_aliases()}")

    # -------------------------------------------------------------------
    # 5. Inspect the pointer backend — entries are pointer files, not zips
    # -------------------------------------------------------------------

    entries_dir = catalog_dir / "entries"
    for p in sorted(entries_dir.iterdir()):
        click.echo(f"\n  {p.name}")
        if p.suffix == POINTER_SUFFIX:
            click.echo(f"    {p.read_text().strip()}")

    # -------------------------------------------------------------------
    # 6. Retrieve an entry by alias and execute it
    # -------------------------------------------------------------------

    entry = catalog.get_catalog_entry("delays-by-origin", maybe_alias=True)
    click.echo(f"\nEntry kind     : {entry.kind}")
    click.echo(f"Entry metadata : {entry.metadata}")

    click.echo("\nDelays by origin:")
    click.echo(entry.expr.execute())

    # -------------------------------------------------------------------
    # 7. Simulate a second user: clone the catalog and fetch content from S3
    # -------------------------------------------------------------------

    _td2 = tempfile.TemporaryDirectory()
    _tmpdirs.append(_td2)
    clone_dir = Path(_td2.name) / "s3-catalog-clone"
    # clone_from auto-detects the pointer backend from the committed content_store.yaml
    clone = Catalog.clone_from(
        url=str(catalog_dir),
        repo_path=clone_dir,
    )
    click.echo(f"\nCloned catalog  : {clone_dir}")
    click.echo(f"Clone entries   : {clone.list()}")
    click.echo(f"Clone aliases   : {clone.list_aliases()}")

    clone_entry = clone.get_catalog_entry("delays-by-carrier", maybe_alias=True)
    click.echo("\nDelays by carrier (from clone):")
    click.echo(clone_entry.expr.execute())

    for td in _tmpdirs:
        td.cleanup()


if __name__ in ("__main__", "__pytest_main__"):
    main()
    pytest_examples_passed = True
