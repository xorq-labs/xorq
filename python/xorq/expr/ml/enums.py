try:
    from enum import StrEnum
except ImportError:
    from strenum import StrEnum


class KVField(StrEnum):
    KEY = "key"
    VALUE = "value"


class ResponseMethod(StrEnum):
    """Sklearn scorer response methods."""

    PREDICT = "predict"
    PREDICT_PROBA = "predict_proba"
    DECISION_FUNCTION = "decision_function"
