from pathlib import Path

import xorq.api as xo
from xorq.catalog.catalog import Catalog
from xorq.catalog.tests.conftest import compare_repo_and_catalog


def test_catalog_compose(repo_cloned_bare, tmpdir):
    cloned = Catalog.clone_from(
        repo_cloned_bare.working_dir, Path(tmpdir).joinpath("cloned")
    )
    expr = next(catalog_entry.expr for catalog_entry in cloned.catalog_entries)
    (on, *_) = expr.columns
    values = expr[on].execute().values
    n = 2
    new_expr = xo.memtable(
        {
            on: [value for value in values for _ in range(n)],
            "new-col": range(n * len(values)),
        }
    )
    joined = expr.join(new_expr, predicates=on)
    expected = joined.execute()
    catalog_entry = cloned.add(joined)
    actual = catalog_entry.expr.execute()
    assert actual.equals(expected)
    compare_repo_and_catalog(repo_cloned_bare, cloned)
