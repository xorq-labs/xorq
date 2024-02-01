import typing as t

from sqlglot.planner import (
    Plan,
    Step,
    Scan,
    Aggregate,
    SetOperation,
    Join,
    name_sequence,
    alias,
    Sort,
)
import sqlglot.expressions as exp


class LetSQLScan(Scan):
    def __init__(self) -> None:
        super().__init__()
        self.source: t.Optional[exp.Expression] = None
        self.target = None

    @classmethod
    def from_expression(
        cls,
        expression: exp.Expression,
        ctes: t.Optional[t.Dict[str, Step]] = None,
        tables: dict = None,
    ) -> Step:
        table = expression
        alias_ = expression.alias_or_name

        if isinstance(expression, exp.Subquery):
            table = expression.this
            step = Step.from_expression(table, ctes)
            step.name = alias_
            return step

        step = LetSQLScan()
        step.name = alias_
        step.source = expression
        if ctes and table.name in ctes:
            step.add_dependency(ctes[table.name])

        if tables and table.name in tables:
            step.target = tables[table.name].target
        else:
            step.target = "LetSQL (local)"

        return step

    def _to_s(self, indent: str) -> t.List[str]:
        return [f"{indent}Source: {self.source.sql() if self.source else '-static-'} {self.target}"]  # type: ignore


class LetSQLStep(Step):
    @classmethod
    def from_expression(
        cls,
        expression: exp.Expression,
        ctes: t.Optional[t.Dict[str, Step]] = None,
        tables: dict = None,
    ) -> Step:
        """
        Builds a DAG of Steps from a SQL expression so that it's easier to execute in an engine.
        Note: the expression's tables and subqueries must be aliased for this method to work. For
        example, given the following expression:

        SELECT
          x.a,
          SUM(x.b)
        FROM x AS x
        JOIN y AS y
          ON x.a = y.a
        GROUP BY x.a

        the following DAG is produced (the expression IDs might differ per execution):

        - Aggregate: x (4347984624)
            Context:
              Aggregations:
                - SUM(x.b)
              Group:
                - x.a
            Projections:
              - x.a
              - "x".""
            Dependencies:
            - Join: x (4347985296)
              Context:
                y:
                On: x.a = y.a
              Projections:
              Dependencies:
              - Scan: x (4347983136)
                Context:
                  Source: x AS x
                Projections:
              - Scan: y (4343416624)
                Context:
                  Source: y AS y
                Projections:

        Args:
            expression: the expression to build the DAG from.
            ctes: a dictionary that maps CTEs to their corresponding Step DAG by name.
            tables: a dictionary that maps table names to the corresponding target system

        Returns:
            A Step DAG corresponding to `expression`.
        """
        ctes = ctes or {}
        expression = expression.unnest()
        with_ = expression.args.get("with")

        # CTEs break the mold of scope and introduce themselves to all in the context.
        if with_:
            ctes = ctes.copy()
            for cte in with_.expressions:
                step = LetSQLStep.from_expression(cte.this, ctes, tables=tables)
                step.name = cte.alias
                ctes[step.name] = step  # type: ignore

        from_ = expression.args.get("from")

        if isinstance(expression, exp.Select) and from_:
            step = LetSQLScan.from_expression(from_.this, ctes, tables=tables)
        elif isinstance(expression, exp.Union):
            step = SetOperation.from_expression(expression, ctes)
        else:
            step = LetSQLScan()

        joins = expression.args.get("joins")

        if joins:
            join = Join.from_joins(joins, ctes)
            join.name = step.name
            join.add_dependency(step)
            step = join

        projections = []  # final selects in this chain of steps representing a select
        operands = {}  # intermediate computations of agg funcs eg x + 1 in SUM(x + 1)
        aggregations = set()
        next_operand_name = name_sequence("_a_")

        def extract_agg_operands(expression):
            agg_funcs = tuple(expression.find_all(exp.AggFunc))
            if agg_funcs:
                aggregations.add(expression)

            for agg in agg_funcs:
                for operand in agg.unnest_operands():
                    if isinstance(operand, exp.Column):
                        continue
                    if operand not in operands:
                        operands[operand] = next_operand_name()

                    operand.replace(exp.column(operands[operand], quoted=True))

            return bool(agg_funcs)

        def set_ops_and_aggs(step):
            step.operands = tuple(
                alias(operand, alias_) for operand, alias_ in operands.items()
            )
            step.aggregations = list(aggregations)

        for e in expression.expressions:
            if e.find(exp.AggFunc):
                projections.append(exp.column(e.alias_or_name, step.name, quoted=True))
                extract_agg_operands(e)
            else:
                projections.append(e)

        where = expression.args.get("where")

        if where:
            step.condition = where.this

        group = expression.args.get("group")

        if group or aggregations:
            aggregate = Aggregate()
            aggregate.source = step.name
            aggregate.name = step.name

            having = expression.args.get("having")

            if having:
                if extract_agg_operands(exp.alias_(having.this, "_h", quoted=True)):
                    aggregate.condition = exp.column("_h", step.name, quoted=True)
                else:
                    aggregate.condition = having.this

            set_ops_and_aggs(aggregate)

            # give aggregates names and replace projections with references to them
            aggregate.group = {
                f"_g{i}": e for i, e in enumerate(group.expressions if group else [])
            }

            intermediate: t.Dict[str | exp.Expression, str] = {}
            for k, v in aggregate.group.items():
                intermediate[v] = k
                if isinstance(v, exp.Column):
                    intermediate[v.name] = k

            for projection in projections:
                for node, *_ in projection.walk():
                    name = intermediate.get(node)
                    if name:
                        node.replace(exp.column(name, step.name))

            if aggregate.condition:
                for node, *_ in aggregate.condition.walk():
                    name = intermediate.get(node) or intermediate.get(node.name)
                    if name:
                        node.replace(exp.column(name, step.name))

            aggregate.add_dependency(step)
            step = aggregate

        order = expression.args.get("order")

        if order:
            if isinstance(step, Aggregate):
                for i, ordered in enumerate(order.expressions):
                    if extract_agg_operands(
                        exp.alias_(ordered.this, f"_o_{i}", quoted=True)
                    ):
                        ordered.this.replace(
                            exp.column(f"_o_{i}", step.name, quoted=True)
                        )

                set_ops_and_aggs(aggregate)

            sort = Sort()
            sort.name = step.name
            sort.key = order.expressions
            sort.add_dependency(step)
            step = sort

        step.projections = projections

        if isinstance(expression, exp.Select) and expression.args.get("distinct"):
            distinct = Aggregate()
            distinct.source = step.name
            distinct.name = step.name
            distinct.group = {
                e.alias_or_name: exp.column(col=e.alias_or_name, table=step.name)
                for e in projections or expression.expressions
            }
            distinct.add_dependency(step)
            step = distinct

        limit = expression.args.get("limit")

        if limit:
            step.limit = int(limit.text("expression"))

        return step


class LetSQLPlan(Plan):
    def __init__(self, expression: exp.Expression, tables: dict = None) -> None:
        super().__init__(expression)
        self.expression = expression.copy()
        self.root = LetSQLStep.from_expression(self.expression, tables=tables)
        self._dag: t.Dict[Step, t.Set[Step]] = {}
