from sqlglot import expressions as exp


class LetSQLPredict(exp.Func):
    arg_types = {"this": True, "expressions": True}
