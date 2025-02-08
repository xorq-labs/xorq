from letsql.ibis_yaml.translate import translate_from_yaml, translate_to_yaml


class IbisYamlCompiler:
    def __init__(self):
        pass

    def compile_to_yaml(self, expr):
        return translate_to_yaml(expr.op(), self)

    def compile_from_yaml(self, yaml_dict):
        return translate_from_yaml(yaml_dict, self)
