from xorq.tests.util import assert_frame_equal


def test_multiple_execution_letsql_register_table(ls_con, csv_dir):
    table_name = "csv_players"
    t = ls_con.read_csv(csv_dir / "awards_players.csv", table_name=table_name)
    ls_con.register(t, table_name=f"{ls_con.name}_{table_name}")

    assert (first := t.execute()) is not None
    assert (second := t.execute()) is not None
    assert_frame_equal(first, second)
