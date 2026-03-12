"""Utilities for inspecting and remapping column references in sklearn pipelines.

The only sklearn components that hold column name references are:
  - ColumnTransformer  (primary — triple (name, estimator, cols) structure)
  - HistGradientBoosting{Classifier,Regressor}  (categorical_features param)

All other transformers (OneHotEncoder, SimpleImputer, StandardScaler, …) are
column-agnostic — they operate on whatever array ColumnTransformer routes to them.

Column map keys are fully path-qualified, mirroring how the tree is traversed:
  - Pipeline and FeatureUnion step names contribute a path segment with trailing slash
  - ColumnTransformer slot names are the terminal segment (no trailing slash)

Example: Pipeline([("prep", ColumnTransformer([("num", ..., cols), ...]))])
  → slot "num" is addressed as "prep/num"
  → nested slot inside a sub-pipeline "prep/num/inner_ct/imputed"
"""

from __future__ import annotations

from functools import cached_property

from attr import field, frozen
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
)
from sklearn.pipeline import FeatureUnion, Pipeline


# Column-param holders: estimator class → param name holding column references.
# ColumnTransformer is excluded — its triple structure is handled separately.
_COLUMN_PARAMS: tuple[tuple[type, str], ...] = (
    (HistGradientBoostingClassifier, "categorical_features"),
    (HistGradientBoostingRegressor, "categorical_features"),
)


# ---------------------------------------------------------------------------
# ColumnRemapper
# ---------------------------------------------------------------------------


@frozen
class ColumnRemapper:
    """Inspect and remap column references in sklearn pipelines.

    column_map_tuple: tuple of (path, new_col_tuple) pairs.
        Keys are fully path-qualified slot names, e.g. "preprocessor/num".
        Slots absent from the map are left unchanged.
        Non-list/tuple selectors (slice, int index, callable) are always left unchanged.
    extra_registry: additional (EstimatorClass, param_name) pairs appended to
        the built-in _COLUMN_PARAMS registry.

    Prefer constructing via ColumnRemapper.from_dict for ergonomics.
    """

    column_map_tuple: tuple[tuple[str, tuple[str, ...]], ...] = field(factory=tuple)
    extra_registry: tuple[tuple[type, str], ...] = field(factory=tuple)

    @cached_property
    def column_map(self) -> dict[str, list[str]]:
        return {k: list(v) for k, v in self.column_map_tuple}

    @cached_property
    def _all_params(self) -> tuple[tuple[type, str], ...]:
        return _COLUMN_PARAMS + self.extra_registry

    def remap(self, pipeline, *, strict=False):
        """Return a new pipeline with column_map applied. No mutation.

        strict: if True, raise if any pipeline path is absent from column_map.
        """
        existing = set(self.list_column_refs(pipeline))
        unknown = set(self.column_map) - existing
        if unknown:
            raise ValueError(
                f"column_map keys not found in pipeline: {sorted(unknown)}\n"
                f"available paths: {sorted(existing)}"
            )
        if strict:
            unmapped = existing - set(self.column_map)
            if unmapped:
                raise ValueError(
                    f"pipeline paths not covered by column_map: {sorted(unmapped)}\n"
                    f"add these keys or disable strict mode"
                )
        return _remap_estimator(pipeline, self.column_map, self._all_params)

    @staticmethod
    def list_column_refs(pipeline) -> dict[str, list[str]]:
        """Return {path: col_list} for every ColumnTransformer slot."""
        return {
            f"{path}{name}": list(cols)
            for ct, path in _walk_cts(pipeline)
            for name, _, cols in ct.transformers
            if isinstance(cols, (list, tuple))
        }

    @classmethod
    def from_dict(cls, column_map: dict[str, list[str]], **kwargs):
        return cls(
            column_map_tuple=tuple((k, tuple(v)) for k, v in column_map.items()),
            **kwargs,
        )


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------


def _clone_with(estimator, **overrides):
    """New estimator instance with the same params, except those in overrides."""
    return estimator.__class__(**{**estimator.get_params(deep=False), **overrides})


def _remap_estimator(estimator, column_map, param_registry, path=""):
    """Recursively reconstruct estimator with column references remapped."""
    match estimator:
        case Pipeline():
            return _clone_with(
                estimator,
                steps=[
                    (
                        name,
                        _remap_estimator(
                            est, column_map, param_registry, f"{path}{name}/"
                        ),
                    )
                    for name, est in estimator.steps
                ],
            )
        case FeatureUnion():
            return _clone_with(
                estimator,
                transformer_list=[
                    (
                        name,
                        _remap_estimator(
                            est, column_map, param_registry, f"{path}{name}/"
                        ),
                    )
                    for name, est in estimator.transformer_list
                ],
            )
        case ColumnTransformer():
            return _clone_with(
                estimator,
                transformers=[
                    (
                        name,
                        _remap_estimator(
                            est, column_map, param_registry, f"{path}{name}/"
                        ),
                        _remap_cols(f"{path}{name}", cols, column_map),
                    )
                    for name, est, cols in estimator.transformers
                ],
            )
        case _:
            for cls, param in param_registry:
                if isinstance(estimator, cls):
                    match getattr(estimator, param, None):
                        case list() if f"{path}{param}" in column_map:
                            return _clone_with(
                                estimator, **{param: list(column_map[f"{path}{param}"])}
                            )
            return clone(estimator)


def _remap_cols(path: str, cols, column_map: dict):
    match cols:
        case _ if path in column_map:
            return list(column_map[path])
        case _:
            return cols


def _walk_cts(estimator, path=""):
    """Depth-first generator yielding (ColumnTransformer, path_prefix) pairs."""
    match estimator:
        case Pipeline():
            for name, est in estimator.steps:
                yield from _walk_cts(est, f"{path}{name}/")
        case FeatureUnion():
            for name, est in estimator.transformer_list:
                yield from _walk_cts(est, f"{path}{name}/")
        case ColumnTransformer():
            yield estimator, path
            for name, est, _ in estimator.transformers:
                yield from _walk_cts(est, f"{path}{name}/")


# ---------------------------------------------------------------------------
# ParamRemapper
# ---------------------------------------------------------------------------


@frozen
class ParamRemapper:
    """Remap arbitrary parameter values in sklearn pipelines.

    param_map_tuple: tuple of (sklearn_path, value) pairs.
        Keys use sklearn's double-underscore convention, e.g. "classifier__C".
        A key matching a step/slot name exactly replaces the whole estimator.
        A key with further segments replaces the named param on the leaf estimator.
        Valid paths are discoverable via pipeline.get_params(deep=True).

    Prefer constructing via ParamRemapper.from_dict for ergonomics.
    """

    param_map_tuple: tuple[tuple[str, object], ...] = field(factory=tuple)

    @cached_property
    def param_map(self) -> dict[str, object]:
        return dict(self.param_map_tuple)

    def remap(self, pipeline, *, strict=False):
        """Return a new pipeline with param_map applied. No mutation.

        strict: if True, raise if any param_map key is absent from
            pipeline.get_params(deep=True).
        """
        known = set(pipeline.get_params(deep=True))
        unknown = set(self.param_map) - known
        if unknown:
            raise ValueError(
                f"param_map keys not found in pipeline: {sorted(unknown)}\n"
                f"available paths: {sorted(known)}"
            )
        if strict:
            unmapped = known - set(self.param_map)
            if unmapped:
                raise ValueError(
                    f"pipeline params not covered by param_map: {sorted(unmapped)}\n"
                    f"add these keys or disable strict mode"
                )
        return _resolve_param_overrides(pipeline, self.param_map)

    @classmethod
    def from_dict(cls, param_map: dict[str, object]):
        return cls(param_map_tuple=tuple(param_map.items()))


def _resolve_param_overrides(estimator, param_map: dict):
    """Recursively reconstruct estimator with param_map applied. No mutation.

    param_map keys use sklearn's __ convention. At each container node the
    leading segment is consumed and the remainder passed to the child.
    An exact match on a step/slot name replaces the whole child estimator.
    """
    if not param_map:
        return clone(estimator)

    match estimator:
        case Pipeline():
            return _clone_with(
                estimator,
                steps=[
                    (name, _resolve_child(est, name, param_map))
                    for name, est in estimator.steps
                ],
            )
        case FeatureUnion():
            return _clone_with(
                estimator,
                transformer_list=[
                    (name, _resolve_child(est, name, param_map))
                    for name, est in estimator.transformer_list
                ],
            )
        case ColumnTransformer():
            return _clone_with(
                estimator,
                transformers=[
                    (name, _resolve_child(est, name, param_map), cols)
                    for name, est, cols in estimator.transformers
                ],
            )
        case _:
            return _clone_with(estimator, **param_map)


def _resolve_child(est, name: str, param_map: dict):
    """Return the (possibly replaced/updated) child estimator for a named slot."""
    if name in param_map:
        return param_map[name]
    sub_map = {
        k[len(name) + 2 :]: v for k, v in param_map.items() if k.startswith(f"{name}__")
    }
    return _resolve_param_overrides(est, sub_map)
