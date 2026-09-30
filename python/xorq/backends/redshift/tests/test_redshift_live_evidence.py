"""Live evidence: the op inventory's classifications and the re-asked
leader-node acceptances, each held as a test against compute.

Runs only with ``-m redshift`` and the ``XORQ_REDSHIFT_*`` variables; see the
conftest beside this file for the harness rules.
"""

from __future__ import annotations

import pytest

import xorq.common.exceptions as com
import xorq.vendor.ibis as ibis
from xorq.backends.redshift.compiler import compiler as redshift_compiler
from xorq.backends.redshift.tests.redshift_live_corpus import (
    INHERITED,
    RECHECK,
    RENAMED,
    compile_inherited,
)
from xorq.backends.redshift.tests.redshift_live_harness import (
    PROBE_TABLE_SCHEMA,
    refuse_unless_compute_select,
)
from xorq.tests.test_redshift_op_inventory import (
    DIALECT_RENAMED,
    INVENTORY,
    PRESENT,
)


# Corpus entries whose op the compiler refuses at compile time. Refusing is a
# measured decision each time: the op reaches something Redshift lacks, or a
# type the mapper refuses.
REFUSED_AT_COMPILE = frozenset(
    {
        "corr",
        "pg_typeof",
        "timezone",
        "array_remove",
        "cardinality",
        "generate_series",
        "map",
        "akeys",
        "avals",
        "exist",
        "jsonb_extract_path_text",
        "row",
    }
)


@pytest.mark.parametrize("name", sorted(set(INHERITED) - REFUSED_AT_COMPILE))
def test_inherited_function_runs_on_compute(name, probe_table, run_select):
    run_select(compile_inherited(name, probe_table, redshift_compiler))


@pytest.mark.parametrize("name", sorted(REFUSED_AT_COMPILE))
def test_inherited_function_is_refused_at_compile(name):
    table = ibis.table(PROBE_TABLE_SCHEMA, name="t", database="s")
    with pytest.raises(
        (
            com.OperationNotDefinedError,
            com.UnsupportedOperationError,
            com.UnsupportedBackendType,
        )
    ):
        compile_inherited(name, table, redshift_compiler)


@pytest.mark.parametrize("name", sorted(RENAMED))
def test_dialect_renamed_function_runs_on_compute(name, probe_table, run_select):
    run_select(compile_inherited(name, probe_table, redshift_compiler, RENAMED))


def test_every_present_classification_has_a_live_probe():
    """A PRESENT entry in the op inventory must be backed by a probe here."""
    probed = {name.split("/")[0] for name in (*INHERITED, *RENAMED)}
    # ``split`` and ``array_size`` are the names the visitors ask for; the
    # dialect renders them as the DIALECT_RENAMED names, which RENAMED probes.
    measured_elsewhere = {"bool_or", "length", "max", "split", "array_size"}
    unbacked = sorted(
        name
        for name, (tag, _) in (*INVENTORY.items(), *DIALECT_RENAMED.items())
        if tag == PRESENT and name not in probed | measured_elsewhere
    )
    assert not unbacked, f"PRESENT with no live probe: {unbacked}"


def test_recheck_escape_decoding(live_config, run_select):
    table = live_config["XORQ_REDSHIFT_PROBE_TABLE"]
    ((length, decoded),) = run_select(RECHECK["tab_is_decoded"].format(table=table))
    assert (length, decoded) == (1, "DECODED")


def test_recheck_regex_backslash_classes(live_config, run_select):
    table = live_config["XORQ_REDSHIFT_PROBE_TABLE"]
    ((digits, total),) = run_select(
        RECHECK["regex_doubled_backslash_is_a_digit_class"].format(table=table)
    )
    assert digits == total
    ((digit_hits, letter_hits, total),) = run_select(
        RECHECK["regex_single_backslash_d_is_the_letter"].format(table=table)
    )
    assert (digit_hits, letter_hits) == (0, total)


@pytest.mark.parametrize(
    "name",
    sorted(
        set(RECHECK)
        - {
            "tab_is_decoded",
            "regex_doubled_backslash_is_a_digit_class",
            "regex_single_backslash_d_is_the_letter",
        }
    ),
)
def test_recheck_runs_on_compute(name, live_config, run_select):
    run_select(RECHECK[name].format(table=live_config["XORQ_REDSHIFT_PROBE_TABLE"]))


@pytest.mark.parametrize(
    "sql",
    [
        pytest.param("SELECT 1", id="no-table"),
        pytest.param("SELECT 'ab' = 'ab '", id="constant-comparison"),
        pytest.param("SELECT * FROM (SELECT 1 AS x) q", id="derived-constant"),
        pytest.param("DELETE FROM s.t", id="delete"),
        pytest.param("SELECT 1 FROM s.t; DROP TABLE s.t", id="two-statements"),
        pytest.param("WITH x AS (SELECT 1) UPDATE s.t SET a = 1", id="cte-update"),
    ],
)
def test_the_harness_refuses_leader_node_and_mutating_probes(sql):
    """Needs no warehouse: the guard runs before anything is sent."""
    with pytest.raises(ValueError):
        refuse_unless_compute_select(sql)


def test_the_harness_passes_a_compute_select():
    refuse_unless_compute_select('SELECT COUNT(*) FROM "s"."t" WHERE title = \'SET\'')
