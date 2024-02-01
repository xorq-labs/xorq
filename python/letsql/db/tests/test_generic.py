import letsql.db as db


def test_explain(data_dir, capsys):
    data_path = data_dir / "csv" / "diamonds.csv"
    db.register_csv("diamonds", data_path)

    query = """select x, y, z from diamonds;"""
    db.sql(query).explain()
    captured = capsys.readouterr()
    assert "LetSQLScan" in captured.out
