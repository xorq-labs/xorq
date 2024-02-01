import json
from collections import deque

from sqlglot import case, and_, or_, func
from sqlglot.expressions import Paren, Is, null

from operator import ge, lt


def _is_null(arg):
    return Is(this=arg, expression=null())


def booster2sql(regressor, columns):
    trees = regressor.get_booster().get_dump(dump_format="json")
    mapping = {column.alias_or_name: column for column in columns}
    if len(trees) < 10:
        case_trees = []
        for tree in trees:
            root = json.loads(tree)

            stack = deque()

            stack.append((root, []))

            cases = []
            while stack:
                cursor, expressions = stack.popleft()

                if "leaf" in cursor:  # is a leaf
                    cases.append([and_(*expressions), cursor["leaf"]])
                    continue

                yes_id = cursor["yes"]
                no_id = cursor["no"]
                split = cursor["split"]
                split_condition = cursor["split_condition"]

                yes_expression = (
                    lt(mapping[split], split_condition)
                    if mapping.get(split)
                    else f"{split} < {split_condition}"
                )
                no_expression = or_(
                    _is_null(mapping[split]), ge(mapping[split], split_condition)
                )

                for child in cursor["children"]:
                    if child["nodeid"] == yes_id:
                        stack.appendleft((child, expressions + [yes_expression]))
                    elif child["nodeid"] == no_id:
                        stack.appendleft((child, expressions + [no_expression]))

            case_tree = case()
            for condition, then in cases:
                case_tree = case_tree.when(condition, then)

            case_trees.append(case_tree)

        head, *tail = (Paren(this=exp) for exp in case_trees)
        prediction = 1 / Paren(this=1 + func("exp", -sum(tail, start=head)))
        return prediction
