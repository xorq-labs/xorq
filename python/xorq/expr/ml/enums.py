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
    FEATURE_IMPORTANCES = "feature_importances"


class FittedStepTagKey(StrEnum):
    TRANSFORM = "FittedStep-transform"
    PREDICT = "FittedStep-predict"
    PREDICT_PROBA = "FittedStep-predict_proba"
    DECISION_FUNCTION = "FittedStep-decision_function"
    FEATURE_IMPORTANCES = "FittedStep-feature_importances"


class FittedPipelineTagKey(StrEnum):
    ALL_STEPS = "FittedPipeline-all_steps"
    TRAINING = "FittedPipeline-training"
    TRANSFORM = "FittedPipeline-transform"
    PREDICT = "FittedPipeline-predict"
    PREDICT_PROBA = "FittedPipeline-predict_proba"
    DECISION_FUNCTION = "FittedPipeline-decision_function"
    FEATURE_IMPORTANCES = "FittedPipeline-feature_importances"
