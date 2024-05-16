import pyarrow as pa

from letsql.internal import (
    LogicalPlan,
    OptimizationRule,
    OptimizerRule,
    SessionContext,
    SessionState,
)


class PassThroughOptimizationRule(OptimizationRule):
    def try_optimize(self, plan: LogicalPlan) -> LogicalPlan:
        print("The PassThrough rule was applied")
        return plan


def test_with_pass_through_rule(capsys):
    state = SessionState()
    rule = OptimizerRule(PassThroughOptimizationRule())
    state = state.add_optimizer_rule(rule)
    context = SessionContext(state)
    table = pa.Table.from_pydict({"a": [1, 2, 3], "b": [4, None, 6]})
    context.register_record_batches("t", [table.to_batches()])
    context.sql("select a, b from t").collect()

    captured = capsys.readouterr()
    assert "The PassThrough rule was applied" in captured.out
