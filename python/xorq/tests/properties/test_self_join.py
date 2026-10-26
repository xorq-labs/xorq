import operator

import pandas as pd
from hypothesis import given
from hypothesis import strategies as st

import xorq.api as xo
from xorq.tests.strategies import key_frames


# how `pub` is made from two key columns: a coalesce over an outer, left or
# inner join (a value, not a field, on the chain), or a plain join field
PUB_SOURCES = {
    "outer": lambda a, b: a.outer_join(b, a.a_pub == b.e_pub).select(
        pub=xo._.a_pub.fill_null(xo._.e_pub)
    ),
    "left": lambda a, b: a.left_join(b, a.a_pub == b.e_pub).select(
        pub=xo._.a_pub.fill_null(xo._.e_pub)
    ),
    "inner": lambda a, b: a.join(b, a.a_pub == b.e_pub).select(
        pub=xo._.a_pub.fill_null(xo._.e_pub)
    ),
    "plain": lambda a, b: a.join(b, a.a_pub == b.e_pub).select(pub=a.a_pub),
}
COMPARISONS = (operator.eq, operator.ne, operator.ge, operator.lt)
PREDICATES = {
    "value": lambda j, r: j.pub == r.pub,
    "deferred": lambda j, r: xo._.pub == r.pub,
    "name": lambda j, r: "pub",
    "pair": lambda j, r: [("pub", "pub")],
}

derived_copies = st.tuples(st.sampled_from(COMPARISONS), st.integers(-3, 3))


def pub_chain(how, a_frame, e_frame):
    return PUB_SOURCES[how](xo.memtable(a_frame), xo.memtable(e_frame))


def derived_copy(j, copy):
    compare, threshold = copy
    return j.filter(compare(j.pub, threshold))


def left_merge(left, right, name):
    # SQL equality never matches a null key, so null right keys are dropped
    # before pandas' merge, which would match NaN with NaN
    right = right[["pub"]].dropna().rename(columns={"pub": name})
    return left.merge(right, left_on="pub", right_on=name, how="left")


def canonical(df, columns):
    return (
        df[list(columns)]
        .astype("float64")
        .sort_values(list(columns), na_position="first")
        .reset_index(drop=True)
    )


def assert_same_rows(actual, expected, columns):
    pd.testing.assert_frame_equal(
        canonical(actual, columns), canonical(expected, columns)
    )


# The oracle executes the chain and its derived copy on their own, which never
# binds one to the other, and merges them in pandas. A predicate whose left
# side binds to the copy (ON copy.pub = copy.pub) matches every copy row
# instead, which the merge does not.


@given(
    a_frame=key_frames("a_pub"),
    e_frame=key_frames("e_pub"),
    how=st.sampled_from(sorted(PUB_SOURCES)),
    copy=derived_copies,
    predicate=st.sampled_from(sorted(PREDICATES)),
)
def test_left_join_to_derived_copy_matches_merge(
    a_frame, e_frame, how, copy, predicate
):
    j = pub_chain(how, a_frame, e_frame)
    r = derived_copy(j, copy)
    actual = j.left_join(r, PREDICATES[predicate](j, r)).execute()
    expected = left_merge(j.execute(), r.execute(), "pub_right")
    assert_same_rows(actual, expected, ["pub", "pub_right"])


@given(
    a_frame=key_frames("a_pub"),
    e_frame=key_frames("e_pub"),
    how=st.sampled_from(sorted(PUB_SOURCES)),
    copy=derived_copies,
    predicate=st.sampled_from(["value", "name"]),
)
def test_select_after_join_to_derived_copy_matches_merge(
    a_frame, e_frame, how, copy, predicate
):
    j = pub_chain(how, a_frame, e_frame)
    r = derived_copy(j, copy)
    expr = j.left_join(r, PREDICATES[predicate](j, r)).select("pub", "pub_right")
    expected = left_merge(j.execute(), r.execute(), "pub_right")
    assert_same_rows(expr.execute(), expected, ["pub", "pub_right"])


@given(
    a_frame=key_frames("a_pub"),
    e_frame=key_frames("e_pub"),
    how=st.sampled_from(sorted(PUB_SOURCES)),
    # two equal filters are one node, and joining a chain to a table it already
    # holds needs .view(): ambiguous by construction, on main as well
    copies=st.lists(derived_copies, min_size=2, max_size=2, unique=True),
)
def test_select_after_two_derived_copies_matches_merge(a_frame, e_frame, how, copies):
    j = pub_chain(how, a_frame, e_frame)
    r, s = (derived_copy(j, copy) for copy in copies)
    q = j.left_join(r, j.pub == r.pub)
    expr = q.left_join(s, q.pub == s.pub, rname="{name}_second").select(
        "pub", "pub_right", "pub_second"
    )
    expected = left_merge(
        left_merge(j.execute(), r.execute(), "pub_right"),
        s.execute(),
        "pub_second",
    )
    assert_same_rows(expr.execute(), expected, ["pub", "pub_right", "pub_second"])
