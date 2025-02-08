from letsql.ibis_yaml.translate import translate_from_yaml, translate_to_yaml


class IbisYamlCompiler:
    def __init__(self):
        pass

    def compile_to_yaml(self, expr):
        self.current_relation = None
        unbound_expr = expr.unbind()
        return translate_to_yaml(unbound_expr.op(), self)

    def compile_from_yaml(self, yaml_dict):
        self.current_relation = None
        return translate_from_yaml(yaml_dict, self)
