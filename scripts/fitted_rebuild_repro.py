"""Minimal repro for catalog replay --rebuild of a FittedPipeline entry.

Adds training + scoring memtables, an identity transform, and a
FittedPipeline predict expression to a source catalog, then tries to
replay the source catalog into a fresh target with ``rebuild=True``.

The rebuild fails during ``catalog.add`` of the rebuilt predict entry
because the predict-chain subtree contains a Read op whose ``hash_path``
is relative (``memtables/<hash>.parquet``), and ``normalize_read`` at
``dask_normalize_expr.py:360`` raises when the path doesn't match any
supported scheme and doesn't resolve as an existing file.

Expected outcome: ``REBUILD FAILED: NotImplementedError: Don't know how
to deal with path "memtables/<hash>.parquet"``.
"""

import tempfile
from pathlib import Path

import sklearn.linear_model
import sklearn.pipeline
import sklearn.preprocessing

import xorq.api as xo
from xorq.catalog.backend import GitBackend
from xorq.catalog.bind import bind
from xorq.catalog.catalog import Catalog
from xorq.catalog.replay import Replayer
from xorq.vendor.ibis.expr import operations as ops


tmp = Path(tempfile.mkdtemp())
src_repo = Catalog.init_repo_path(tmp.joinpath("src"))
src = Catalog(backend=GitBackend(repo=src_repo))

training = src.add(
    xo.memtable({"f": [1.0, 2.0, 3.0, 4.0], "t": [0.0, 0.0, 1.0, 1.0]}),
    aliases=("training",),
)
scoring = src.add(
    xo.memtable({"f": [1.5, 2.5], "t": [0.0, 1.0]}),
    aliases=("scoring",),
)
unbound = ops.UnboundTable(name="p", schema=training.expr.schema()).to_expr()
identity = src.add(unbound.select("f", "t"), aliases=("identity",))

sk_pipeline = sklearn.pipeline.make_pipeline(
    sklearn.preprocessing.StandardScaler(),
    sklearn.linear_model.LinearRegression(),
)
pipeline = xo.Pipeline.from_instance(sk_pipeline)
fitted = pipeline.fit(bind(training, identity), features=("f",), target="t")
predict_expr = fitted.predict(bind(scoring, identity))
preds = src.add(predict_expr, aliases=("preds",))
print(f"added preds: {preds.name}")

target = Catalog.from_repo_path(tmp.joinpath("tgt"), init=True)
try:
    # Replayer(from_catalog=src, rebuild=True).replay(target)
    replayer = Replayer(from_catalog=src, rebuild=True)
    replayer.replay(target)
    # *_, op = replayer.ops
    # self = replayer
    # source_entry = replayer.from_catalog.get_catalog_entry(op.entry_hash)
    # from xorq.catalog.replay import _rebuild_subexpr
    # expr = _rebuild_subexpr(
    #     source_entry.lazy_expr,
    #     from_catalog=replayer.from_catalog,
    #     to_catalog=target,
    #     remap={},
    # )

    print("REBUILD SUCCEEDED")
except Exception as e:
    print(f"REBUILD FAILED: {type(e).__name__}: {str(e)[:400]}")
