import ibis
import pytest
from packaging import version
from pytest import param

from xorq.common.utils.ibis_utils import from_ibis


def test_string_concat():
    s1 = ibis.literal("hello")
    s2 = ibis.literal("world")
    expr = s1 + s2

    assert from_ibis(expr) is not None


def test_string_upper_lower():
    s = ibis.literal("Hello")
    upper_expr = s.upper()
    lower_expr = s.lower()

    assert from_ibis(upper_expr) is not None
    assert from_ibis(lower_expr) is not None


def test_string_to_date():
    value = "20170206"
    format_str = "%Y%m%d"
    s = ibis.literal(value)
    expr = s.as_date(format_str)

    assert from_ibis(expr) is not None


def test_string_length():
    s = ibis.literal("hello")
    expr = s.length()

    assert from_ibis(expr) is not None


def test_string_substring():
    s = ibis.literal("hello world")
    expr = s.substr(0, 5)

    assert from_ibis(expr) is not None


def test_string_startswith():
    t = ibis.table({"id": "int32", "name": "string"}, name="T")
    q = t.filter(t.name.startswith("X"))

    assert from_ibis(q) is not None


def test_string_endswith():
    t = ibis.table({"id": "int32", "name": "string"}, name="T")
    q = t.filter(t.name.endswith("X"))

    assert from_ibis(q) is not None


def cases(t):
    args = [(1, "abcd"), (2, "ABCD")]
    kwargs = {"else_": "dabc"}
    if version.parse(ibis.__version__) <= version.parse("9.5.0"):
        args = [[(1, "abcd"), (2, "ABCD")]]
        kwargs = {"default": "dabc"}

    return t.int_col.cases(*args, **kwargs).startswith("abc")


@pytest.mark.parametrize(
    "fun",
    [
        param(
            lambda t: t.string_col.contains("6"),
            id="contains",
        ),
        param(
            lambda t: t.string_col.rlike("|".join(map(str, range(10)))),
            id="rlike",
        ),
        param(
            lambda t: ("a" + t.string_col + "a").re_search(r"\d+"),
            id="re_search_substring",
        ),
        param(
            lambda t: t.string_col.re_search(r"\d+"),
            id="re_search",
        ),
        param(
            lambda t: t.string_col.re_search(r"[[:digit:]]+"),
            id="re_search_posix",
        ),
        param(
            lambda t: t.string_col.re_extract(r"([[:digit:]]+)", 1),
            id="re_extract_posix",
        ),
        param(
            lambda t: (t.string_col + "1").re_extract(r"\d(\d+)", 0),
            id="re_extract_whole_group",
        ),
        param(
            lambda t: t.date_string_col.re_extract(r"(\d+)\D(\d+)\D(\d+)", 1),
            id="re_extract_group_1",
        ),
        param(
            lambda t: t.date_string_col.re_extract(r"(\d+)\D(\d+)\D(\d+)", 2),
            id="re_extract_group_2",
        ),
        param(
            lambda t: t.date_string_col.re_extract(r"(\d+)\D(\d+)\D(\d+)", 3),
            id="re_extract_group_3",
        ),
        param(
            lambda t: t.date_string_col.re_extract(r"^(\d+)", 1),
            id="re_extract_group_at_beginning",
        ),
        param(
            lambda t: t.date_string_col.re_extract(r"(\d+)$", 1),
            id="re_extract_group_at_end",
        ),
        param(
            lambda t: t.string_col.re_replace(r"[[:digit:]]+", "a"),
            id="re_replace_posix",
        ),
        param(
            lambda t: t.string_col.re_replace(r"\d+", "a"),
            id="re_replace",
        ),
        param(
            lambda t: t.string_col.repeat(2),
            id="repeat_method",
        ),
        param(
            lambda t: 2 * t.string_col,
            id="repeat_left",
        ),
        param(
            lambda t: t.string_col * 2,
            id="repeat_right",
        ),
        param(
            lambda t: t.string_col.translate("01", "ab"),
            id="translate",
        ),
        param(
            lambda t: t.string_col.find("a"),
            id="find",
        ),
        param(
            lambda t: t.date_string_col.find("13", 3),
            id="find_start",
        ),
        param(
            lambda t: t.string_col.lpad(10, "a"),
            id="lpad",
        ),
        param(
            lambda t: t.string_col.rpad(10, "a"),
            id="rpad",
        ),
        param(
            lambda t: t.string_col.lower(),
            id="lower",
        ),
        param(
            lambda t: t.string_col.upper(),
            id="upper",
        ),
        param(
            lambda t: t.string_col.reverse(),
            id="reverse",
        ),
        param(
            lambda t: t.string_col.ascii_str(),
            id="ascii_str",
        ),
        param(
            lambda t: t.string_col.length(),
            id="length",
        ),
        param(
            cases,
            id="startswith",
        ),
        param(
            lambda t: t.date_string_col.startswith("2010-01"),
            id="startswith-simple",
        ),
        param(
            lambda t: t.string_col.strip(),
            id="strip",
        ),
        param(
            lambda t: t.string_col.lstrip(),
            id="lstrip",
        ),
        param(
            lambda t: t.string_col.rstrip(),
            id="rstrip",
        ),
        param(
            lambda t: t.string_col.capitalize(),
            id="capitalize",
        ),
        param(
            lambda t: t.date_string_col.substr(2, 3),
            id="substr",
        ),
        param(
            lambda t: t.date_string_col.substr(2),
            id="substr-start-only",
        ),
        param(
            lambda t: t.date_string_col.left(2),
            id="left",
        ),
        param(
            lambda t: t.date_string_col.right(2),
            id="right",
        ),
        param(
            lambda t: t.date_string_col[1:3],
            id="slice",
        ),
        param(
            lambda t: t.date_string_col[-2],
            id="negative-index",
        ),
        param(
            lambda t: t.date_string_col[: t.date_string_col.length()],
            id="expr_slice_end",
        ),
        param(
            lambda t: t.date_string_col[:],
            id="expr_empty_slice",
        ),
        param(
            lambda t: t.date_string_col[
                t.date_string_col.length() - 2 : t.date_string_col.length() - 1
            ],
            id="expr_slice_begin_end",
        ),
        param(
            lambda t: ibis.literal("-").join(["a", t.string_col, "c"]),
            id="join",
        ),
        param(
            lambda t: t.string_col + t.date_string_col,
            id="concat_columns",
        ),
        param(
            lambda t: t.string_col + "a",
            id="concat_column_scalar",
        ),
        param(
            lambda t: "a" + t.string_col,
            id="concat_scalar_column",
        ),
        param(
            lambda t: t.string_col.replace("1", "42"),
            id="replace",
        ),
    ],
)
def test_string(ibis_alltypes, fun):
    expr = fun(ibis_alltypes).name("tmp")

    assert from_ibis(expr).execute() is not None
