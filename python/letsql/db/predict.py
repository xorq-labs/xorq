from sqlglot import expressions as exp


class LetSQLPredict(exp.Func):
    arg_types = {"this": True, "expressions": True}
    _sql_names = ["PREDICT_XGB"]
