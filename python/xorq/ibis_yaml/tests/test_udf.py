import pandas as pd
import pyarrow as pa
import pytest

import xorq.api as xo
import xorq.expr.datatypes as dt
import xorq.expr.udf as udf
import xorq.ibis_yaml
import xorq.ibis_yaml.utils
import xorq.vendor.ibis.expr.operations as ops
from xorq.common.utils.dasher import tokenize
from xorq.common.utils.graph_utils import walk_nodes
from xorq.expr.udf import pyarrow_udwf
from xorq.vendor import ibis


def test_built_in_udf_properties(compiler):
    t = xo.table({"a": "int64"}, name="t")

    @xo.udf.scalar.builtin
    def add_one(x: int) -> int:
        return x + 1

    expr = t.mutate(new=add_one(t.a))
    yaml_dict = compiler.to_yaml(expr)
    roundtrip_expr = compiler.from_yaml(yaml_dict)

    original_mutation = expr.op()
    roundtrip_mutation = roundtrip_expr.op()

    original_udf = original_mutation.values["new"]
    roundtrip_udf = roundtrip_mutation.values["new"]

    assert original_udf.__func_name__ == roundtrip_udf.__func_name__
    assert original_udf.__input_type__ == roundtrip_udf.__input_type__
    assert original_udf.dtype == roundtrip_udf.dtype
    assert len(original_udf.args) == len(roundtrip_udf.args)

    for orig_arg, rt_arg in zip(original_udf.args, roundtrip_udf.args):
        assert orig_arg.dtype == rt_arg.dtype


def test_compiler_raises(compiler):
    t = xo.table({"a": "int64"}, name="t")

    @xo.udf.scalar.python
    def add_one(x: int) -> int:
        pass

    expr = t.mutate(new=add_one(t.a))
    with pytest.raises(NotImplementedError):
        compiler.to_yaml(expr)


@pytest.mark.xfail(
    reason="UDFs do not have the same memory address when pickled/unpickled"
)
def test_built_in_udf(compiler):
    # (Pdb) diffs[3][2].args[0] == diffs[3][1].args[0]
    # False
    # (Pdb) diffs[3][2].args[0]
    # <ibis.expr.operations.relations.Project object at 0x7ffff48f53d0>
    # (Pdb) diffs[3][2].args[0].args
    # (<ibis.expr.operations.relations.UnboundTable object at 0x7ffff48f5310>, {'a': <ibis.expr.operations.relations.Field object at 0x7ffff490cbb0>, 'new': <tests.test_udf.add_one_1 object at 0x7ffff48f5550>})
    # (Pdb) diffs[3][1].args[0].args
    # (<ibis.expr.operations.relations.UnboundTable object at 0x7ffff48f45f0>, {'a': <ibis.expr.operations.relations.Field object at 0x7ffff490cb40>, 'new': <tests.test_udf.add_one_1 object at 0x7ffff48f49b0>})
    t = xo.table({"a": "int64"}, name="t")

    @xo.udf.scalar.builtin
    def add_one(x: int) -> int:
        pass

    expr = t.mutate(new=add_one(t.a))
    yaml_dict = compiler.to_yaml(expr)
    roundtrip_expr = compiler.from_yaml(yaml_dict)
    print(f"Original {expr}")
    print(f"Roundtrip {roundtrip_expr}")
    xorq.ibis_yaml.utils.diff_ibis_exprs(expr, roundtrip_expr)

    assert roundtrip_expr.equals(expr)


def test_pandas_udf_properties(compiler):
    t = xo.table({"a": "int64", "b": "float64"}, name="t")

    @xo.udf.make_pandas_udf(
        schema=xo.schema({"a": int, "b": float}),
        return_type=xo.expr.datatypes.float64,
        name="multiply_add",
    )
    def multiply_add(df):
        return df["a"] * df["b"] + df["a"]

    expr = t.mutate(result=multiply_add.on_expr(t))
    yaml_dict = compiler.to_yaml(expr)

    roundtrip_expr = compiler.from_yaml(yaml_dict)

    original_mutation = expr.op()
    roundtrip_mutation = roundtrip_expr.op()
    original_udf = original_mutation.values["result"]
    roundtrip_udf = roundtrip_mutation.values["result"]

    assert original_udf.__func_name__ == roundtrip_udf.__func_name__
    assert original_udf.__input_type__ == roundtrip_udf.__input_type__
    assert original_udf.dtype == roundtrip_udf.dtype
    assert len(original_udf.args) == len(roundtrip_udf.args)


@pytest.fixture
def df():
    batch = pa.RecordBatch.from_arrays(
        [
            pa.array([0, 1, 2, 3, 4, 5, 6]),
            pa.array([7, 4, 3, 8, 9, 1, 6]),
            pa.array(["A", "A", "A", "A", "B", "B", "B"]),
        ],
        names=["a", "b", "c"],
    )

    return batch.to_pandas()


def test_udwf_roundtrip(compiler, df):
    @pyarrow_udwf(
        schema=ibis.schema({"a": float}),
        return_type=ibis.dtype(float),
        alpha=0.9,
    )
    def exp_smooth(self, values: list[pa.Array], num_rows: int) -> pa.Array:
        results = []
        curr_value = 0.0
        values = values[0]
        for idx in range(num_rows):
            if idx == 0:
                curr_value = values[idx].as_py()
            else:
                curr_value = values[idx].as_py() * self.alpha + curr_value * (
                    1.0 - self.alpha
                )
            results.append(curr_value)

        return pa.array(results)

    con = xo.connect()
    t = con.register(df, table_name="t")

    expr = t.select(
        t.a,
        udwf=exp_smooth.on_expr(t).over(ibis.window(group_by=ibis._.c)),
    ).order_by(t.a)

    yaml_dict = compiler.to_yaml(expr)

    profiles = {con._profile.hash_name: con}
    roundtrip_expr = compiler.from_yaml(yaml_dict, profiles=profiles)

    roundtrip_expr.execute()


def test_aggudf_func_token_survives_roundtrip(compiler):
    con = xo.connect()
    t = con.register(pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [4.0, 5.0, 6.0]}), "t")

    @udf.agg.pandas_df(
        schema=ibis.schema({"a": dt.float64, "b": dt.float64}),
        return_type=dt.float64,
        name="mysum",
    )
    def mysum(sample_df):
        return float(sample_df["a"].sum() + sample_df["b"].sum())

    expr = t.mutate(s=mysum.on_expr(t))

    def aggudf_func_tokens(e):
        return sorted(
            tokenize(node.__func__)
            for node in walk_nodes(object, e)
            if isinstance(node, ops.udf.AggUDF)
        )

    live_tokens = aggudf_func_tokens(expr)
    assert len(live_tokens) == 1

    profiles = {con._profile.hash_name: con}
    roundtrip = compiler.from_yaml(compiler.to_yaml(expr), profiles=profiles)
    assert aggudf_func_tokens(roundtrip) == live_tokens
