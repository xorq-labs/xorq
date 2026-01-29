import pandas as pd
import pytest

import xorq.api as xo
import xorq.expr.datatypes as dt
from xorq.expr.ml.structer import (
    KV_ENCODED_TYPE,
    KVEncoder,
    KVField,
    Structer,
)


sklearn = pytest.importorskip("sklearn")


class TestKVField:
    """Pedantic tests for KVField StrEnum behavior."""

    def test_kvfield_is_strenum(self):
        """Test KVField inherits from StrEnum."""
        try:
            from enum import StrEnum
        except ImportError:
            from strenum import StrEnum
        assert issubclass(KVField, StrEnum)

    def test_kvfield_members_are_strings(self):
        """Test KVField members are string instances."""
        assert isinstance(KVField.KEY, str)
        assert isinstance(KVField.VALUE, str)

    def test_kvfield_string_equality(self):
        """Test KVField members equal their string values."""
        assert KVField.KEY == "key"
        assert KVField.VALUE == "value"

    def test_kvfield_membership_by_member(self):
        """Test membership check works with enum members."""
        assert KVField.KEY in KVField
        assert KVField.VALUE in KVField

    def test_kvfield_callable_lookup(self):
        """Test KVField can be called with string value to get member."""
        assert KVField("key") is KVField.KEY
        assert KVField("value") is KVField.VALUE
        with pytest.raises(ValueError):
            KVField("nonexistent")

    def test_kvfield_value_attribute(self):
        """Test .value attribute returns the string value."""
        assert KVField.KEY.value == "key"
        assert KVField.VALUE.value == "value"

    def test_kvfield_name_attribute(self):
        """Test .name attribute returns the member name."""
        assert KVField.KEY.name == "KEY"
        assert KVField.VALUE.name == "VALUE"

    def test_kvfield_iteration(self):
        """Test iterating over KVField yields all members."""
        members = list(KVField)
        assert len(members) == 2
        assert KVField.KEY in members
        assert KVField.VALUE in members

    def test_kvfield_as_dict_key(self):
        """Test KVField members work as dictionary keys."""
        d = {KVField.KEY: "key_value", KVField.VALUE: "value_value"}
        assert d["key"] == "key_value"
        assert d["value"] == "value_value"
        assert d[KVField.KEY] == "key_value"
        assert d[KVField.VALUE] == "value_value"

    def test_kvfield_string_operations(self):
        """Test KVField members support string operations."""
        assert KVField.KEY.upper() == "KEY"
        assert KVField.VALUE.capitalize() == "Value"
        assert KVField.KEY + "_suffix" == "key_suffix"

    def test_kvfield_hash_matches_string(self):
        """Test KVField members hash the same as their string values."""
        assert hash(KVField.KEY) == hash("key")
        assert hash(KVField.VALUE) == hash("value")

    def test_kv_encoded_type_uses_kvfield_values(self):
        """Test KV_ENCODED_TYPE struct uses KVField.value strings as keys."""
        struct_fields = KV_ENCODED_TYPE.value_type.fields
        # Keys are plain strings (for YAML serialization compatibility)
        assert "key" in struct_fields
        assert "value" in struct_fields
        # KVField members also work due to string equality
        assert KVField.KEY in struct_fields
        assert KVField.VALUE in struct_fields


class TestKVEncoder:
    """Tests for KVEncoder encode/decode functionality."""

    def test_encode_basic(self):
        """Test KVEncoder.encode with a simple transformer."""
        from sklearn.preprocessing import OneHotEncoder

        df = pd.DataFrame({"cat": ["a", "b", "a"]})
        model = OneHotEncoder(sparse_output=False)
        model.fit(df)

        result = KVEncoder.encode(model, df)

        assert isinstance(result, pd.Series)
        assert len(result) == 3
        # Each element should be a tuple of dicts with 'key' and 'value'
        first_row = result.iloc[0]
        assert all("key" in d and "value" in d for d in first_row)

    def test_encode_feature_names(self):
        """Test that encode uses get_feature_names_out() for keys."""
        from sklearn.preprocessing import OneHotEncoder

        df = pd.DataFrame({"cat": ["a", "b", "c"]})
        model = OneHotEncoder(sparse_output=False)
        model.fit(df)

        result = KVEncoder.encode(model, df)
        keys = [d["key"] for d in result.iloc[0]]

        expected_names = list(model.get_feature_names_out())
        assert keys == expected_names

    def test_decode_basic(self):
        """Test KVEncoder.decode expands series to individual columns."""
        series = pd.Series(
            [
                ({"key": "a", "value": 1.0}, {"key": "b", "value": 0.0}),
                ({"key": "a", "value": 0.0}, {"key": "b", "value": 1.0}),
            ]
        )

        result = KVEncoder.decode(series)

        assert "a" in result.columns
        assert "b" in result.columns
        assert result["a"].tolist() == [1.0, 0.0]
        assert result["b"].tolist() == [0.0, 1.0]

    def test_decode_empty_series_raises(self):
        """Test decode raises ValueError for empty series."""
        series = pd.Series([], dtype=object)
        with pytest.raises(ValueError):
            KVEncoder.decode(series)

    def test_is_kv_encoded_type_true(self):
        """Test is_kv_encoded_type returns True for KV format."""
        assert KVEncoder.is_kv_encoded_type(KV_ENCODED_TYPE)

    def test_is_kv_encoded_type_false(self):
        """Test is_kv_encoded_type returns False for non-KV types."""
        assert not KVEncoder.is_kv_encoded_type(dt.float64)
        assert not KVEncoder.is_kv_encoded_type(dt.Array(dt.float64))
        assert not KVEncoder.is_kv_encoded_type(dt.Struct({"a": dt.float64}))


class TestStructer:
    """Tests for Structer class."""

    def test_from_names_typ(self):
        """Test creating Structer with known schema."""
        structer = Structer.from_names_typ(("a", "b"), dt.float64)

        assert not structer.is_kv_encoded
        assert structer.struct is not None
        assert "a" in structer.struct.fields
        assert "b" in structer.struct.fields

    def test_kv_encoded_factory(self):
        """Test creating KV-encoded Structer."""
        structer = Structer.kv_encoded(input_columns=("x", "y"))

        assert structer.is_kv_encoded
        assert structer.struct is None
        assert structer.input_columns == ("x", "y")
        assert structer.return_type == KV_ENCODED_TYPE

    def test_needs_target_default_false(self):
        """Test needs_target defaults to False."""
        structer = Structer.from_names_typ(("a",), dt.float64)
        assert structer.needs_target is False

    def test_needs_target_explicit(self):
        """Test needs_target can be set explicitly."""
        base = Structer.from_names_typ(("a",), dt.float64)
        structer = Structer(struct=base.struct, needs_target=True)
        assert structer.needs_target is True

    def test_is_series_default_false(self):
        """Test is_series defaults to False."""
        structer = Structer.from_names_typ(("a",), dt.float64)
        assert structer.is_series is False

    def test_is_series_explicit(self):
        """Test is_series can be set explicitly."""
        structer = Structer(struct=None, input_columns=("text",), is_series=True)
        assert structer.is_series is True

    def test_return_type_struct(self):
        """Test return_type for known schema."""
        structer = Structer.from_names_typ(("a", "b"), dt.float64)
        assert isinstance(structer.return_type, dt.Struct)

    def test_return_type_kv_encoded(self):
        """Test return_type for KV-encoded."""
        structer = Structer.kv_encoded(input_columns=("x",))
        assert structer.return_type == KV_ENCODED_TYPE

    def test_output_columns_struct(self):
        """Test output_columns for known schema."""
        structer = Structer.from_names_typ(("a", "b"), dt.float64)
        assert structer.output_columns == ("a", "b")

    def test_output_columns_kv_encoded_raises(self):
        """Test output_columns raises for KV-encoded."""
        structer = Structer.kv_encoded(input_columns=("x",))
        with pytest.raises(ValueError, match="KV-encoded"):
            _ = structer.output_columns

    def test_get_convert_array_kv_encoded_raises(self):
        """Test get_convert_array raises for KV-encoded."""
        structer = Structer.kv_encoded(input_columns=("x",))
        with pytest.raises(ValueError, match="KV-encoded"):
            structer.get_convert_array()

    def test_get_output_columns_known_schema(self):
        """Test get_output_columns returns dtype keys for known schema."""
        structer = Structer.from_names_typ(("a", "b"), dt.float64)
        assert structer.get_output_columns() == ("a", "b")
        # dest_col is ignored for known schema
        assert structer.get_output_columns(dest_col="foo") == ("a", "b")

    def test_get_output_columns_kv_encoded(self):
        """Test get_output_columns returns dest_col tuple for KV-encoded."""
        structer = Structer.kv_encoded(input_columns=("x",))
        assert structer.get_output_columns() == ("transformed",)
        assert structer.get_output_columns(dest_col="encoded") == ("encoded",)

    def test_maybe_unpack_known_schema(self):
        """Test maybe_unpack unpacks struct column for known schema."""
        from sklearn.preprocessing import StandardScaler

        t = xo.memtable({"a": [1.0, 2.0], "b": [3.0, 4.0]})
        step = xo.Step.from_instance_name(StandardScaler(), name="scaler")
        fitted = step.fit(t, features=("a", "b"))

        # Transform returns struct column that needs unpacking
        result = fitted.transform(t)
        df = result.execute()

        # Should have unpacked columns, not a struct column
        assert "a" in df.columns
        assert "b" in df.columns
        assert "transformed" not in df.columns

    def test_maybe_unpack_kv_encoded(self):
        """Test maybe_unpack returns expr unchanged for KV-encoded."""
        from sklearn.preprocessing import OneHotEncoder

        t = xo.memtable({"cat": ["a", "b", "a"]})
        step = xo.Step.from_instance_name(OneHotEncoder(), name="ohe")
        fitted = step.fit(t, features=("cat",))

        # Transform returns KV-encoded column (not unpacked)
        result = fitted.transform(t)
        df = result.execute()

        # Should have KV-encoded column, not unpacked
        assert "transformed" in df.columns
        # The column contains tuples of dicts
        assert all("key" in d and "value" in d for d in df["transformed"].iloc[0])


class TestStructerFromInstance:
    """Tests for structer_from_instance dispatch."""

    def test_standard_scaler(self):
        """Test StandardScaler produces known schema."""
        from sklearn.preprocessing import StandardScaler

        t = xo.memtable({"a": [1.0, 2.0], "b": [3.0, 4.0]})
        structer = Structer.from_instance_expr(StandardScaler(), t, features=("a", "b"))

        assert not structer.is_kv_encoded
        assert not structer.needs_target
        assert not structer.is_series

    def test_simple_imputer(self):
        """Test SimpleImputer produces known schema."""
        from sklearn.impute import SimpleImputer

        t = xo.memtable({"a": [1.0, None], "b": [3.0, 4.0]})
        structer = Structer.from_instance_expr(SimpleImputer(), t, features=("a", "b"))

        assert not structer.is_kv_encoded
        assert not structer.needs_target

    def test_one_hot_encoder(self):
        """Test OneHotEncoder produces KV-encoded schema."""
        from sklearn.preprocessing import OneHotEncoder

        t = xo.memtable({"cat": ["a", "b", "c"]})
        structer = Structer.from_instance_expr(OneHotEncoder(), t, features=("cat",))

        assert structer.is_kv_encoded
        assert not structer.needs_target
        assert not structer.is_series

    def test_select_k_best(self):
        """Test SelectKBest has needs_target=True."""
        from sklearn.feature_selection import SelectKBest

        t = xo.memtable({"a": [1.0, 2.0], "b": [3.0, 4.0]})
        structer = Structer.from_instance_expr(SelectKBest(k=1), t, features=("a", "b"))

        assert not structer.is_kv_encoded
        assert structer.needs_target is True

    def test_tfidf_vectorizer(self):
        """Test TfidfVectorizer is KV-encoded and is_series."""
        from sklearn.feature_extraction.text import TfidfVectorizer

        t = xo.memtable({"text": ["hello world", "foo bar"]})
        structer = Structer.from_instance_expr(TfidfVectorizer(), t, features=("text",))

        assert structer.is_kv_encoded
        assert structer.is_series
        assert not structer.needs_target

    def test_unregistered_raises(self):
        """Test unregistered transformer raises ValueError."""

        class CustomTransformer:
            pass

        t = xo.memtable({"a": [1.0, 2.0]})
        with pytest.raises(ValueError, match="can't handle type"):
            Structer.from_instance_expr(CustomTransformer(), t)


class TestKVEncoderIntegration:
    """Integration tests for KV encoding with xorq."""

    def test_one_hot_encoder_step(self):
        """Test OneHotEncoder through Step interface."""
        from sklearn.preprocessing import OneHotEncoder

        t = xo.memtable({"cat": ["a", "b", "a", "c"]})
        step = xo.Step.from_instance_name(OneHotEncoder(), name="ohe")
        fitted = step.fit(t, features=("cat",))

        result = fitted.transform(t)
        df = result.execute()

        # Should have KV-encoded column
        assert "transformed" in df.columns

        # Decode and verify
        decoded = KVEncoder.decode(df["transformed"])
        assert "cat_a" in decoded.columns
        assert "cat_b" in decoded.columns
        assert "cat_c" in decoded.columns

    def test_tfidf_vectorizer_step(self):
        """Test TfidfVectorizer through Step interface."""
        from sklearn.feature_extraction.text import TfidfVectorizer

        t = xo.memtable({"text": ["hello world", "foo bar", "hello foo"]})
        step = xo.Step.from_instance_name(TfidfVectorizer(), name="tfidf")
        fitted = step.fit(t, features=("text",))

        result = fitted.transform(t)
        df = result.execute()

        # Should have KV-encoded column
        assert "transformed" in df.columns

        # Decode and verify we get vocabulary columns
        decoded = KVEncoder.decode(df["transformed"])
        assert "hello" in decoded.columns
        assert "world" in decoded.columns
        assert "foo" in decoded.columns
        assert "bar" in decoded.columns

    def test_tfidf_matches_sklearn(self):
        """Test TfidfVectorizer output matches sklearn."""
        from sklearn.feature_extraction.text import TfidfVectorizer

        texts = ["hello world", "foo bar baz", "hello foo"]
        t = xo.memtable({"text": texts})

        # xorq result
        step = xo.Step.from_instance_name(TfidfVectorizer(), name="tfidf")
        fitted = step.fit(t, features=("text",))
        result = fitted.transform(t)
        df = result.execute()
        xorq_decoded = KVEncoder.decode(df["transformed"])

        # sklearn result
        model = TfidfVectorizer()
        sklearn_result = model.fit_transform(texts).toarray()
        sklearn_df = pd.DataFrame(sklearn_result, columns=model.get_feature_names_out())

        # Compare (reorder columns to match)
        xorq_sorted = xorq_decoded[sorted(xorq_decoded.columns)]
        sklearn_sorted = sklearn_df[sorted(sklearn_df.columns)]

        pd.testing.assert_frame_equal(
            xorq_sorted.reset_index(drop=True),
            sklearn_sorted.reset_index(drop=True),
            check_exact=False,
            atol=1e-10,
        )

    def test_select_k_best_with_target(self):
        """Test SelectKBest works with target column."""
        from sklearn.feature_selection import SelectKBest, f_classif

        t = xo.memtable(
            {
                "a": [1.0, 2.0, 3.0, 4.0],
                "b": [4.0, 3.0, 2.0, 1.0],
                "target": [0, 0, 1, 1],
            }
        )

        step = xo.Step.from_instance_name(
            SelectKBest(score_func=f_classif, k=1), name="skb"
        )
        fitted = step.fit(t, features=("a", "b"), target="target")

        result = fitted.transform(t)
        df = result.execute()

        # Should have selected 1 feature
        assert len(df.columns) == 2  # target + 1 selected feature

    def test_get_kv_value_type_non_kv(self):
        """Test get_kv_value_type returns original type for non-KV types."""
        typ = dt.float64
        assert KVEncoder.get_kv_value_type(typ) == dt.float64

    def test_structer_dtype_raises_for_kv_encoded(self):
        """Test dtype property raises for KV-encoded Structer."""
        structer = Structer.kv_encoded(input_columns=("x",))
        with pytest.raises(ValueError, match="KV-encoded"):
            _ = structer.dtype

    def test_select_k_best_mixed_types_raises(self):
        """Test SelectKBest raises for mixed feature types."""
        from sklearn.feature_selection import SelectKBest

        t = xo.memtable({"a": [1.0, 2.0], "b": ["x", "y"]})
        with pytest.raises(ValueError):
            Structer.from_instance_expr(SelectKBest(k=1), t, features=("a", "b"))


class TestColumnTransformerStructer:
    """Tests for ColumnTransformer structer_from_instance registration."""

    def test_known_schema_all_transformers(self):
        """Test ColumnTransformer with all known-schema transformers."""
        from sklearn.compose import ColumnTransformer
        from sklearn.preprocessing import StandardScaler

        t = xo.memtable({"num1": [1.0, 2.0], "num2": [3.0, 4.0]})
        ct = ColumnTransformer(
            [
                ("scaler", StandardScaler(), ["num1", "num2"]),
            ]
        )
        structer = Structer.from_instance_expr(ct, t)

        # is_kv_encoded is for leaf transformers only
        assert not structer.any_kv_encoded
        assert structer.struct is not None
        # sklearn-style prefixes
        assert "scaler__num1" in structer.struct.fields
        assert "scaler__num2" in structer.struct.fields
        assert structer.passthrough_columns == ()

    def test_kv_encoded_transformer(self):
        """Test ColumnTransformer with KV-encoded transformer."""
        from sklearn.compose import ColumnTransformer
        from sklearn.preprocessing import OneHotEncoder

        t = xo.memtable({"cat": ["a", "b", "c"]})
        ct = ColumnTransformer(
            [
                ("encoder", OneHotEncoder(), ["cat"]),
            ]
        )
        structer = Structer.from_instance_expr(ct, t)

        # Containers use any_kv_encoded (has KV column in hybrid struct)
        assert structer.any_kv_encoded
        assert structer.input_columns == ("cat",)
        assert structer.passthrough_columns == ()
        # KV-encoded child produces named KV column
        assert "encoder" in structer.struct.fields

    def test_mixed_known_and_kv_encoded(self):
        """Test ColumnTransformer with mixed known-schema and KV-encoded."""
        from sklearn.compose import ColumnTransformer
        from sklearn.preprocessing import OneHotEncoder, StandardScaler

        t = xo.memtable({"num": [1.0, 2.0], "cat": ["a", "b"]})
        ct = ColumnTransformer(
            [
                ("scaler", StandardScaler(), ["num"]),
                ("encoder", OneHotEncoder(), ["cat"]),
            ]
        )
        structer = Structer.from_instance_expr(ct, t)

        # Hybrid output: struct set with some KV columns
        assert structer.any_kv_encoded
        assert not structer.all_kv_encoded
        # Known-schema column with prefix
        assert "scaler__num" in structer.struct.fields
        # KV-encoded column named after transformer
        assert "encoder" in structer.struct.fields
        # Only cat feeds into KV transformer
        assert structer.input_columns == ("cat",)

    def test_passthrough_explicit(self):
        """Test ColumnTransformer with explicit passthrough."""
        from sklearn.compose import ColumnTransformer
        from sklearn.preprocessing import StandardScaler

        t = xo.memtable({"num": [1.0, 2.0], "cat": ["a", "b"]})
        ct = ColumnTransformer(
            [
                ("scaler", StandardScaler(), ["num"]),
                ("pass", "passthrough", ["cat"]),
            ]
        )
        structer = Structer.from_instance_expr(ct, t)

        assert not structer.any_kv_encoded
        # sklearn-style prefixes
        assert "scaler__num" in structer.struct.fields
        assert "pass__cat" in structer.struct.fields
        assert structer.passthrough_columns == ("pass__cat",)

    def test_remainder_passthrough(self):
        """Test ColumnTransformer with remainder='passthrough'."""
        from sklearn.compose import ColumnTransformer
        from sklearn.preprocessing import StandardScaler

        t = xo.memtable({"num": [1.0, 2.0], "cat": ["a", "b"], "other": [3.0, 4.0]})
        ct = ColumnTransformer(
            [("scaler", StandardScaler(), ["num"])],
            remainder="passthrough",
        )
        structer = Structer.from_instance_expr(ct, t)

        assert not structer.any_kv_encoded
        # sklearn-style prefixes
        assert "scaler__num" in structer.struct.fields
        assert "remainder__cat" in structer.struct.fields
        assert "remainder__other" in structer.struct.fields
        assert set(structer.passthrough_columns) == {
            "remainder__cat",
            "remainder__other",
        }

    def test_remainder_passthrough_with_kv_encoded(self):
        """Test ColumnTransformer with remainder='passthrough' and KV-encoded transformer."""
        from sklearn.compose import ColumnTransformer
        from sklearn.preprocessing import OneHotEncoder

        t = xo.memtable({"cat": ["a", "b"], "num1": [1.0, 2.0], "num2": [3.0, 4.0]})
        ct = ColumnTransformer(
            [("encoder", OneHotEncoder(), ["cat"])],
            remainder="passthrough",
        )
        structer = Structer.from_instance_expr(ct, t)

        assert structer.any_kv_encoded
        assert structer.input_columns == ("cat",)
        # sklearn-style prefixes for remainder
        assert set(structer.passthrough_columns) == {
            "remainder__num1",
            "remainder__num2",
        }

    def test_drop_transformer(self):
        """Test ColumnTransformer with drop transformer."""
        from sklearn.compose import ColumnTransformer
        from sklearn.preprocessing import StandardScaler

        t = xo.memtable({"num": [1.0, 2.0], "cat": ["a", "b"]})
        ct = ColumnTransformer(
            [
                ("scaler", StandardScaler(), ["num"]),
                ("drop", "drop", ["cat"]),
            ]
        )
        structer = Structer.from_instance_expr(ct, t)

        assert not structer.any_kv_encoded
        # sklearn-style prefixes
        assert "scaler__num" in structer.struct.fields
        assert "cat" not in structer.struct.fields
        assert "drop__cat" not in structer.struct.fields

    def test_nested_column_transformer(self):
        """Test nested ColumnTransformer (ColumnTransformer inside ColumnTransformer)."""
        from sklearn.compose import ColumnTransformer
        from sklearn.preprocessing import StandardScaler

        t = xo.memtable({"a": [1.0, 2.0], "b": [3.0, 4.0], "c": [5.0, 6.0]})
        inner_ct = ColumnTransformer(
            [
                ("scaler_inner", StandardScaler(), ["a", "b"]),
            ]
        )
        outer_ct = ColumnTransformer(
            [
                ("inner", inner_ct, ["a", "b"]),
                ("scaler_c", StandardScaler(), ["c"]),
            ]
        )
        structer = Structer.from_instance_expr(outer_ct, t)

        # Both inner and outer have known schemas
        assert not structer.any_kv_encoded
        # Nested sklearn-style prefixes: outer__inner__col
        assert "inner__scaler_inner__a" in structer.struct.fields
        assert "inner__scaler_inner__b" in structer.struct.fields
        assert "scaler_c__c" in structer.struct.fields

    def test_nested_column_transformer_with_kv_encoded(self):
        """Test nested ColumnTransformer where inner has KV-encoded."""
        from sklearn.compose import ColumnTransformer
        from sklearn.preprocessing import OneHotEncoder, StandardScaler

        t = xo.memtable({"cat": ["a", "b"], "num": [1.0, 2.0]})
        inner_ct = ColumnTransformer(
            [
                ("encoder", OneHotEncoder(), ["cat"]),
            ]
        )
        outer_ct = ColumnTransformer(
            [
                ("inner", inner_ct, ["cat"]),
                ("scaler", StandardScaler(), ["num"]),
            ]
        )
        structer = Structer.from_instance_expr(outer_ct, t)

        # Inner is KV-encoded, so outer has KV column
        assert structer.any_kv_encoded
        # Inner CT with KV becomes a single KV column named "inner"
        assert "inner" in structer.struct.fields
        assert "scaler__num" in structer.struct.fields

    def test_unregistered_child_raises(self):
        """Test ColumnTransformer with unregistered child transformer raises."""
        from sklearn.compose import ColumnTransformer
        from sklearn.preprocessing import RobustScaler  # Not registered

        t = xo.memtable({"num": [1.0, 2.0]})
        ct = ColumnTransformer(
            [
                ("robust", RobustScaler(), ["num"]),
            ]
        )

        with pytest.raises(ValueError, match="can't handle type"):
            Structer.from_instance_expr(ct, t)

    # Tests for _normalize_columns (column spec normalization)

    def test_normalize_columns_list(self):
        """Test _normalize_columns handles list -> tuple."""
        from sklearn.compose import ColumnTransformer
        from sklearn.preprocessing import StandardScaler

        t = xo.memtable({"a": [1.0, 2.0], "b": [3.0, 4.0]})
        ct = ColumnTransformer(
            [
                ("scaler", StandardScaler(), ["a", "b"]),  # list
            ]
        )
        structer = Structer.from_instance_expr(ct, t)

        assert not structer.any_kv_encoded
        # sklearn-style prefixes
        assert "scaler__a" in structer.struct.fields
        assert "scaler__b" in structer.struct.fields

    def test_normalize_columns_string(self):
        """Test _normalize_columns handles string -> single-element tuple."""
        from sklearn.compose import ColumnTransformer
        from sklearn.preprocessing import StandardScaler

        t = xo.memtable({"num": [1.0, 2.0], "other": [3.0, 4.0]})
        ct = ColumnTransformer(
            [
                ("scaler", StandardScaler(), "num"),  # string, not list
            ]
        )
        structer = Structer.from_instance_expr(ct, t)

        assert not structer.any_kv_encoded
        # sklearn-style prefix
        assert "scaler__num" in structer.struct.fields

    def test_normalize_columns_tuple(self):
        """Test _normalize_columns handles tuple -> tuple (passthrough)."""
        from sklearn.compose import ColumnTransformer
        from sklearn.preprocessing import StandardScaler

        t = xo.memtable({"a": [1.0, 2.0], "b": [3.0, 4.0]})
        ct = ColumnTransformer(
            [
                ("scaler", StandardScaler(), ("a", "b")),  # tuple, not list
            ]
        )
        structer = Structer.from_instance_expr(ct, t)

        assert not structer.any_kv_encoded
        # sklearn-style prefixes
        assert "scaler__a" in structer.struct.fields
        assert "scaler__b" in structer.struct.fields

    def test_normalize_columns_none(self):
        """Test _normalize_columns handles None -> empty tuple."""
        from sklearn.compose import ColumnTransformer
        from sklearn.preprocessing import StandardScaler

        t = xo.memtable({"a": [1.0, 2.0], "b": [3.0, 4.0]})
        ct = ColumnTransformer(
            [
                ("scaler", StandardScaler(), ["a"]),
                ("pass", "passthrough", None),  # None columns
            ]
        )
        structer = Structer.from_instance_expr(ct, t)

        assert not structer.any_kv_encoded
        # sklearn-style prefix
        assert "scaler__a" in structer.struct.fields

    def test_normalize_columns_invalid_type_raises(self):
        """Test _normalize_columns raises TypeError for unsupported types."""
        from sklearn.compose import ColumnTransformer
        from sklearn.preprocessing import StandardScaler

        t = xo.memtable({"a": [1.0, 2.0]})
        ct = ColumnTransformer(
            [
                ("scaler", StandardScaler(), 123),  # invalid type
            ]
        )

        with pytest.raises(TypeError, match="Unsupported columns type"):
            Structer.from_instance_expr(ct, t)


class TestFeatureUnionStructer:
    """Tests for FeatureUnion structer_from_instance registration."""

    def test_known_schema_all_transformers(self):
        """Test FeatureUnion with all known-schema transformers."""
        from sklearn.impute import SimpleImputer
        from sklearn.pipeline import FeatureUnion
        from sklearn.preprocessing import StandardScaler

        t = xo.memtable({"num1": [1.0, 2.0], "num2": [3.0, 4.0]})
        fu = FeatureUnion(
            [
                ("scaler", StandardScaler()),
                ("imputer", SimpleImputer()),
            ]
        )
        structer = Structer.from_instance_expr(fu, t, features=("num1", "num2"))

        assert not structer.any_kv_encoded
        assert structer.struct is not None
        # sklearn-style prefixes
        assert "scaler__num1" in structer.struct.fields
        assert "scaler__num2" in structer.struct.fields
        assert "imputer__num1" in structer.struct.fields
        assert "imputer__num2" in structer.struct.fields

    def test_kv_encoded_transformer(self):
        """Test FeatureUnion with KV-encoded transformer."""
        from sklearn.pipeline import FeatureUnion
        from sklearn.preprocessing import OneHotEncoder

        t = xo.memtable({"cat": ["a", "b", "c"]})
        fu = FeatureUnion(
            [
                ("encoder", OneHotEncoder()),
            ]
        )
        structer = Structer.from_instance_expr(fu, t, features=("cat",))

        # Containers use any_kv_encoded
        assert structer.any_kv_encoded
        assert structer.input_columns == ("cat",)
        # KV-encoded child produces named KV column
        assert "encoder" in structer.struct.fields

    def test_mixed_known_and_kv_encoded(self):
        """Test FeatureUnion with mixed known-schema and KV-encoded."""
        from sklearn.pipeline import FeatureUnion
        from sklearn.preprocessing import OneHotEncoder, StandardScaler

        t = xo.memtable({"num": [1.0, 2.0], "cat": ["a", "b"]})
        fu = FeatureUnion(
            [
                ("scaler", StandardScaler()),
                ("encoder", OneHotEncoder()),
            ]
        )
        structer = Structer.from_instance_expr(fu, t, features=("num", "cat"))

        # Hybrid output: struct set with some KV columns
        assert structer.any_kv_encoded
        assert not structer.all_kv_encoded
        assert set(structer.input_columns) == {"num", "cat"}
        # Known-schema columns with prefix
        assert "scaler__num" in structer.struct.fields
        assert "scaler__cat" in structer.struct.fields
        # KV-encoded column named after transformer
        assert "encoder" in structer.struct.fields

    def test_unregistered_child_raises(self):
        """Test FeatureUnion with unregistered child transformer raises."""
        from sklearn.pipeline import FeatureUnion
        from sklearn.preprocessing import RobustScaler  # Not registered

        t = xo.memtable({"num": [1.0, 2.0]})
        fu = FeatureUnion(
            [
                ("robust", RobustScaler()),
            ]
        )

        with pytest.raises(ValueError, match="can't handle type"):
            Structer.from_instance_expr(fu, t, features=("num",))

    def test_nested_feature_union(self):
        """Test nested FeatureUnion."""
        from sklearn.pipeline import FeatureUnion
        from sklearn.preprocessing import StandardScaler

        t = xo.memtable({"a": [1.0, 2.0], "b": [3.0, 4.0]})
        inner_fu = FeatureUnion(
            [
                ("scaler1", StandardScaler()),
            ]
        )
        outer_fu = FeatureUnion(
            [
                ("inner", inner_fu),
                ("scaler2", StandardScaler()),
            ]
        )
        structer = Structer.from_instance_expr(outer_fu, t, features=("a", "b"))

        assert not structer.any_kv_encoded
        # Nested sklearn-style prefixes
        assert "inner__scaler1__a" in structer.struct.fields
        assert "inner__scaler1__b" in structer.struct.fields
        assert "scaler2__a" in structer.struct.fields
        assert "scaler2__b" in structer.struct.fields


class TestSklearnPipelineStructer:
    """Tests for sklearn Pipeline structer_from_instance registration."""

    def test_known_schema_pipeline(self):
        """Test Pipeline with all known-schema transformers."""
        from sklearn.impute import SimpleImputer
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler

        t = xo.memtable({"num1": [1.0, 2.0], "num2": [3.0, 4.0]})
        pipe = Pipeline(
            [
                ("imputer", SimpleImputer()),
                ("scaler", StandardScaler()),
            ]
        )
        structer = Structer.from_instance_expr(pipe, t, features=("num1", "num2"))

        assert not structer.any_kv_encoded
        assert structer.struct is not None
        # Pipeline doesn't add prefixes - just chains through
        assert "num1" in structer.struct.fields
        assert "num2" in structer.struct.fields

    def test_kv_encoded_pipeline(self):
        """Test Pipeline with KV-encoded transformer."""
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import OneHotEncoder

        t = xo.memtable({"cat": ["a", "b", "c"]})
        pipe = Pipeline(
            [
                ("encoder", OneHotEncoder()),
            ]
        )
        structer = Structer.from_instance_expr(pipe, t, features=("cat",))

        # Containers use any_kv_encoded
        assert structer.any_kv_encoded
        assert structer.input_columns == ("cat",)
        # Pipeline wraps KV output in "encoded" column
        assert "encoded" in structer.struct.fields

    def test_kv_encoded_early_exit(self):
        """Test Pipeline returns KV-encoded once any step is KV-encoded."""
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import OneHotEncoder

        t = xo.memtable({"cat": ["a", "b"]})
        # Even if there are steps after OneHotEncoder, we return KV-encoded
        pipe = Pipeline(
            [
                ("encoder", OneHotEncoder()),
                # StandardScaler can't actually follow OneHotEncoder in practice,
                # but for schema computation we exit early at KV-encoded
            ]
        )
        structer = Structer.from_instance_expr(pipe, t, features=("cat",))

        assert structer.any_kv_encoded

    def test_nested_pipeline(self):
        """Test nested Pipeline (Pipeline inside Pipeline)."""
        from sklearn.impute import SimpleImputer
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler

        t = xo.memtable({"a": [1.0, 2.0], "b": [3.0, 4.0]})
        inner_pipe = Pipeline(
            [
                ("scaler", StandardScaler()),
            ]
        )
        outer_pipe = Pipeline(
            [
                ("inner", inner_pipe),
                ("imputer", SimpleImputer()),
            ]
        )
        structer = Structer.from_instance_expr(outer_pipe, t, features=("a", "b"))

        assert not structer.any_kv_encoded
        assert "a" in structer.struct.fields
        assert "b" in structer.struct.fields

    def test_pipeline_with_column_transformer(self):
        """Test Pipeline containing ColumnTransformer."""
        from sklearn.compose import ColumnTransformer
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler

        t = xo.memtable({"num": [1.0, 2.0], "cat": ["a", "b"]})
        pipe = Pipeline(
            [
                (
                    "ct",
                    ColumnTransformer(
                        [
                            ("scaler", StandardScaler(), ["num"]),
                        ],
                        remainder="passthrough",
                    ),
                ),
            ]
        )
        structer = Structer.from_instance_expr(pipe, t)

        assert not structer.any_kv_encoded
        # sklearn-style prefixes from CT
        assert "scaler__num" in structer.struct.fields
        assert "remainder__cat" in structer.struct.fields

    def test_pipeline_with_kv_column_transformer(self):
        """Test Pipeline containing ColumnTransformer with KV-encoded child."""
        from sklearn.compose import ColumnTransformer
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import OneHotEncoder

        t = xo.memtable({"cat": ["a", "b"], "num": [1.0, 2.0]})
        pipe = Pipeline(
            [
                (
                    "ct",
                    ColumnTransformer(
                        [
                            ("encoder", OneHotEncoder(), ["cat"]),
                        ],
                        remainder="passthrough",
                    ),
                ),
            ]
        )
        structer = Structer.from_instance_expr(pipe, t)

        # Hybrid output
        assert structer.any_kv_encoded
        # KV column named "encoder", passthrough column with prefix
        assert "encoder" in structer.struct.fields
        assert "remainder__num" in structer.struct.fields

    def test_unregistered_step_raises(self):
        """Test Pipeline with unregistered transformer raises."""
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import RobustScaler  # Not registered

        t = xo.memtable({"num": [1.0, 2.0]})
        pipe = Pipeline(
            [
                ("robust", RobustScaler()),
            ]
        )

        with pytest.raises(ValueError, match="can't handle type"):
            Structer.from_instance_expr(pipe, t, features=("num",))

    def test_empty_pipeline_returns_input_schema(self):
        """Test empty Pipeline returns input schema (passthrough)."""
        from sklearn.pipeline import Pipeline

        t = xo.memtable({"num": [1.0, 2.0]})
        pipe = Pipeline([])

        # Empty pipeline is effectively passthrough
        structer = Structer.from_instance_expr(pipe, t, features=("num",))
        assert not structer.any_kv_encoded
        assert "num" in structer.struct.fields

    def test_passthrough_step(self):
        """Test Pipeline with 'passthrough' step."""
        from sklearn.pipeline import Pipeline

        t = xo.memtable({"num1": [1.0, 2.0], "num2": [3.0, 4.0]})
        pipe = Pipeline(
            [
                ("pass", "passthrough"),
            ]
        )
        structer = Structer.from_instance_expr(pipe, t, features=("num1", "num2"))

        assert not structer.any_kv_encoded
        assert "num1" in structer.struct.fields
        assert "num2" in structer.struct.fields

    def test_passthrough_columns_from_final_step(self):
        """Test passthrough_columns reflects only the final step."""
        from sklearn.compose import ColumnTransformer
        from sklearn.impute import SimpleImputer
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler

        t = xo.memtable({"num": [1.0, 2.0], "cat": ["a", "b"]})
        # ColumnTransformer has passthrough, but SimpleImputer doesn't
        pipe = Pipeline(
            [
                (
                    "ct",
                    ColumnTransformer(
                        [
                            ("scaler", StandardScaler(), ["num"]),
                        ],
                        remainder="passthrough",
                    ),
                ),
                ("imputer", SimpleImputer()),
            ]
        )
        structer = Structer.from_instance_expr(pipe, t)

        assert not structer.any_kv_encoded
        # passthrough_columns is from final step (SimpleImputer), which is empty
        assert structer.passthrough_columns == ()

    def test_nested_pipeline_with_passthrough(self):
        """Test Pipeline(Pipeline()) where inner pipeline has passthrough columns."""
        from sklearn.compose import ColumnTransformer
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler

        t = xo.memtable({"num": [1.0, 2.0], "cat": ["a", "b"]})
        inner_pipe = Pipeline(
            [
                (
                    "ct",
                    ColumnTransformer(
                        [
                            ("scaler", StandardScaler(), ["num"]),
                        ],
                        remainder="passthrough",
                    ),
                ),
            ]
        )
        outer_pipe = Pipeline(
            [
                ("inner", inner_pipe),
            ]
        )
        structer = Structer.from_instance_expr(outer_pipe, t)

        assert not structer.any_kv_encoded
        # sklearn-style prefixes from CT
        assert "scaler__num" in structer.struct.fields
        assert "remainder__cat" in structer.struct.fields

    def test_chained_feature_transforms(self):
        """Test Pipeline where features change between steps."""
        from sklearn.feature_selection import SelectKBest
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler

        t = xo.memtable(
            {
                "a": [1.0, 2.0, 3.0, 4.0],
                "b": [4.0, 3.0, 2.0, 1.0],
                "target": [0, 0, 1, 1],
            }
        )
        pipe = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("selector", SelectKBest(k=1)),
            ]
        )
        structer = Structer.from_instance_expr(pipe, t, features=("a", "b"))

        # SelectKBest reduces to k=1 features
        assert not structer.is_kv_encoded
        assert len(structer.struct.fields) == 1


class TestDecodeEncodedColumns:
    """Tests for KVEncoder.decode_encoded_column and decode_encoded_columns."""

    def test_no_encoded_cols(self):
        """Test passthrough when no encoded columns."""
        df = pd.DataFrame({"a": [1.0, 2.0], "b": [3.0, 4.0]})
        features = ("a", "b")
        encoded_cols = ()

        result_df, result_features = KVEncoder.decode_encoded_columns(
            df, features, encoded_cols
        )

        pd.testing.assert_frame_equal(result_df, df)
        assert result_features == features

    def test_decode_encoded_column(self):
        """Test decoding KV-encoded column."""
        df = pd.DataFrame(
            {
                "encoded": [
                    ({"key": "x", "value": 1.0}, {"key": "y", "value": 2.0}),
                    ({"key": "x", "value": 3.0}, {"key": "y", "value": 4.0}),
                ],
                "other": [10, 20],
            }
        )
        features = ("encoded", "other")
        encoded_cols = ("encoded",)

        result_df, result_features = KVEncoder.decode_encoded_columns(
            df, features, encoded_cols
        )

        assert "encoded" not in result_df.columns
        assert "x" in result_df.columns
        assert "y" in result_df.columns
        assert "other" in result_df.columns
        assert set(result_features) == {"x", "y", "other"}

    def test_missing_encoded_col_raises(self):
        """Test that missing encoded columns raise ValueError."""
        df = pd.DataFrame({"a": [1.0, 2.0]})
        features = ("a",)

        with pytest.raises(ValueError, match="nonexistent not in DataFrame"):
            KVEncoder.decode_encoded_column(df, features, "nonexistent")

    def test_empty_encoded_col_raises(self):
        """Test that empty encoded column raises ValueError."""
        df = pd.DataFrame({"encoded": []})
        features = ("encoded",)

        with pytest.raises(ValueError, match="cannot decode empty column"):
            KVEncoder.decode_encoded_column(df, features, "encoded")


class TestGetSchemaOut:
    """Tests for get_schema_out function with sklearn-style prefixing and hybrid output."""

    def test_column_transformer_basic(self):
        """Test get_schema_out with ColumnTransformer uses sklearn-style prefixes."""
        from sklearn.compose import ColumnTransformer
        from sklearn.preprocessing import StandardScaler

        from xorq.expr.ml.structer import get_schema_out

        t = xo.memtable({"num1": [1.0, 2.0], "num2": [3.0, 4.0]})
        ct = ColumnTransformer(
            [
                ("scaler", StandardScaler(), ["num1", "num2"]),
            ]
        )
        schema = get_schema_out(ct, t)

        # sklearn-style prefixing: name__col
        assert "scaler__num1" in schema.names
        assert "scaler__num2" in schema.names

    def test_column_transformer_passthrough(self):
        """Test get_schema_out with ColumnTransformer passthrough uses prefixes."""
        from sklearn.compose import ColumnTransformer
        from sklearn.preprocessing import StandardScaler

        from xorq.expr.ml.structer import get_schema_out

        t = xo.memtable({"num": [1.0, 2.0], "cat": ["a", "b"]})
        ct = ColumnTransformer(
            [
                ("scaler", StandardScaler(), ["num"]),
                ("pass", "passthrough", ["cat"]),
            ]
        )
        schema = get_schema_out(ct, t)

        assert "scaler__num" in schema.names
        assert "pass__cat" in schema.names

    def test_column_transformer_remainder_passthrough(self):
        """Test get_schema_out with ColumnTransformer remainder='passthrough'."""
        from sklearn.compose import ColumnTransformer
        from sklearn.preprocessing import StandardScaler

        from xorq.expr.ml.structer import get_schema_out

        t = xo.memtable({"num": [1.0, 2.0], "cat": ["a", "b"], "other": [3.0, 4.0]})
        ct = ColumnTransformer(
            [("scaler", StandardScaler(), ["num"])],
            remainder="passthrough",
        )
        schema = get_schema_out(ct, t, features=("num", "cat", "other"))

        assert "scaler__num" in schema.names
        # remainder columns get 'remainder__' prefix
        assert "remainder__cat" in schema.names
        assert "remainder__other" in schema.names

    def test_column_transformer_drop(self):
        """Test get_schema_out with ColumnTransformer drop."""
        from sklearn.compose import ColumnTransformer
        from sklearn.preprocessing import StandardScaler

        from xorq.expr.ml.structer import get_schema_out

        t = xo.memtable({"num": [1.0, 2.0], "cat": ["a", "b"]})
        ct = ColumnTransformer(
            [
                ("scaler", StandardScaler(), ["num"]),
                ("drop", "drop", ["cat"]),
            ]
        )
        schema = get_schema_out(ct, t)

        assert "scaler__num" in schema.names
        # Dropped columns should not appear
        assert "cat" not in schema.names
        assert "drop__cat" not in schema.names

    def test_feature_union(self):
        """Test get_schema_out with FeatureUnion uses prefixes."""
        from sklearn.impute import SimpleImputer
        from sklearn.pipeline import FeatureUnion
        from sklearn.preprocessing import StandardScaler

        from xorq.expr.ml.structer import get_schema_out

        t = xo.memtable({"num1": [1.0, 2.0], "num2": [3.0, 4.0]})
        fu = FeatureUnion(
            [
                ("scaler", StandardScaler()),
                ("imputer", SimpleImputer()),
            ]
        )
        schema = get_schema_out(fu, t, features=("num1", "num2"))

        # FeatureUnion also uses prefixes
        assert "scaler__num1" in schema.names
        assert "scaler__num2" in schema.names
        assert "imputer__num1" in schema.names
        assert "imputer__num2" in schema.names

    def test_sklearn_pipeline(self):
        """Test get_schema_out with sklearn Pipeline (no prefixes for pipeline)."""
        from sklearn.impute import SimpleImputer
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler

        from xorq.expr.ml.structer import get_schema_out

        t = xo.memtable({"num1": [1.0, 2.0], "num2": [3.0, 4.0]})
        pipe = Pipeline(
            [
                ("imputer", SimpleImputer()),
                ("scaler", StandardScaler()),
            ]
        )
        schema = get_schema_out(pipe, t, features=("num1", "num2"))

        # Pipeline doesn't add prefixes - just chains through
        assert "num1" in schema.names
        assert "num2" in schema.names

    def test_base_estimator(self):
        """Test get_schema_out with simple BaseEstimator (no prefix)."""
        from sklearn.preprocessing import StandardScaler

        from xorq.expr.ml.structer import get_schema_out

        t = xo.memtable({"a": [1.0, 2.0], "b": [3.0, 4.0]})
        scaler = StandardScaler()
        schema = get_schema_out(scaler, t, features=("a", "b"))

        # Leaf estimators don't add prefixes
        assert "a" in schema.names
        assert "b" in schema.names

    def test_nested_column_transformer(self):
        """Test get_schema_out with nested ColumnTransformer uses nested prefixes."""
        from sklearn.compose import ColumnTransformer
        from sklearn.preprocessing import StandardScaler

        from xorq.expr.ml.structer import get_schema_out

        t = xo.memtable({"a": [1.0, 2.0], "b": [3.0, 4.0], "c": [5.0, 6.0]})
        inner_ct = ColumnTransformer(
            [
                ("scaler_inner", StandardScaler(), ["a", "b"]),
            ]
        )
        outer_ct = ColumnTransformer(
            [
                ("inner", inner_ct, ["a", "b"]),
                ("scaler_c", StandardScaler(), ["c"]),
            ]
        )
        schema = get_schema_out(outer_ct, t)

        # Nested prefixes: outer__inner__col
        assert "inner__scaler_inner__a" in schema.names
        assert "inner__scaler_inner__b" in schema.names
        assert "scaler_c__c" in schema.names

    def test_pipeline_with_column_transformer(self):
        """Test get_schema_out with Pipeline containing ColumnTransformer."""
        from sklearn.compose import ColumnTransformer
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler

        from xorq.expr.ml.structer import get_schema_out

        t = xo.memtable({"num": [1.0, 2.0], "cat": ["a", "b"]})
        pipe = Pipeline(
            [
                (
                    "ct",
                    ColumnTransformer(
                        [
                            ("scaler", StandardScaler(), ["num"]),
                        ],
                        remainder="passthrough",
                    ),
                ),
            ]
        )
        schema = get_schema_out(pipe, t, features=("num", "cat"))

        # Pipeline chains through, CT adds prefixes
        assert "scaler__num" in schema.names
        assert "remainder__cat" in schema.names

    def test_unsupported_type_raises(self):
        """Test get_schema_out raises for unsupported types."""
        from xorq.expr.ml.structer import get_schema_out

        t = xo.memtable({"a": [1.0, 2.0]})

        with pytest.raises(ValueError, match="can't handle type"):
            get_schema_out("invalid", t)

    def test_kv_encoded_deeply_nested_pipeline(self):
        """Test get_schema_out with hybrid output - known-schema + KV-encoded columns.

        Pipeline structure:
        - ColumnTransformer (hybrid output)
          - FeatureUnion (known-schema: numeric features)
            - Pipeline (SimpleImputer -> StandardScaler)
            - Pipeline (SimpleImputer -> StandardScaler)
          - Pipeline (KV-encoded: categorical via OneHotEncoder)
        """
        import numpy as np
        import pandas as pd
        from sklearn.compose import ColumnTransformer
        from sklearn.impute import SimpleImputer
        from sklearn.pipeline import FeatureUnion
        from sklearn.pipeline import Pipeline as SklearnPipeline
        from sklearn.preprocessing import OneHotEncoder, StandardScaler

        from xorq.expr.ml.structer import KV_ENCODED_TYPE, get_schema_out

        np.random.seed(42)
        n_samples = 20

        data = pd.DataFrame(
            {
                "age": np.random.randint(18, 80, n_samples).astype(float),
                "income": np.random.randint(20000, 150000, n_samples).astype(float),
                "education": np.random.choice(
                    ["high_school", "bachelor", "master"], n_samples
                ),
                "region": np.random.choice(["north", "south", "east"], n_samples),
            }
        )

        numeric_features = ["age", "income"]
        categorical_features = ["education", "region"]
        all_features = tuple(numeric_features + categorical_features)

        # Build nested sklearn pipeline
        scaled_pipeline = SklearnPipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]
        )

        imputed_pipeline = SklearnPipeline(
            [
                ("imputer", SimpleImputer(strategy="mean")),
                ("scaler", StandardScaler()),
            ]
        )

        numeric_union = FeatureUnion(
            [
                ("scaled", scaled_pipeline),
                ("imputed", imputed_pipeline),
            ]
        )

        categorical_pipeline = SklearnPipeline(
            [
                ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
                (
                    "encoder",
                    OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                ),
            ]
        )

        preprocessor = ColumnTransformer(
            [
                ("numeric", numeric_union, numeric_features),
                ("categorical", categorical_pipeline, categorical_features),
            ]
        )

        expr = xo.memtable(data)
        schema = get_schema_out(preprocessor, expr, features=all_features)

        # Hybrid output: known-schema numeric columns with sklearn-style prefixes
        # FeatureUnion: numeric__scaled__age, numeric__scaled__income, numeric__imputed__age, etc.
        assert "numeric__scaled__age" in schema.names
        assert "numeric__scaled__income" in schema.names
        assert "numeric__imputed__age" in schema.names
        assert "numeric__imputed__income" in schema.names

        # KV-encoded categorical gets single column named after transformer
        assert "categorical" in schema.names
        assert schema["categorical"] == KV_ENCODED_TYPE

    def test_non_kv_deeply_nested_pipeline(self):
        """Test get_schema_out with depth-4 nested pipeline with all known-schema transformers.

        Pipeline structure:
        - ColumnTransformer (known schema - no KV-encoded children)
          - Pipeline (SimpleImputer -> StandardScaler -> Pipeline)
            - Pipeline (SimpleImputer -> StandardScaler)
          - Pipeline (SimpleImputer -> StandardScaler)
        """
        import numpy as np
        import pandas as pd
        from sklearn.compose import ColumnTransformer
        from sklearn.impute import SimpleImputer
        from sklearn.pipeline import Pipeline as SklearnPipeline
        from sklearn.preprocessing import StandardScaler

        from xorq.expr.ml.structer import get_schema_out

        np.random.seed(42)
        n_samples = 20

        data = pd.DataFrame(
            {
                "age": np.random.randint(18, 80, n_samples).astype(float),
                "income": np.random.randint(20000, 150000, n_samples).astype(float),
                "credit_score": np.random.randint(300, 850, n_samples).astype(float),
                "years_employed": np.random.randint(0, 40, n_samples).astype(float),
                "debt_ratio": np.random.uniform(0, 1, n_samples),
                "savings": np.random.randint(0, 100000, n_samples).astype(float),
            }
        )

        numeric_features_a = ["age", "income", "credit_score"]
        numeric_features_b = ["years_employed", "debt_ratio", "savings"]
        all_features = tuple(numeric_features_a + numeric_features_b)

        # Build nested sklearn pipeline (depth 4)
        inner_pipeline = SklearnPipeline(
            [
                ("imputer2", SimpleImputer(strategy="mean")),
                ("scaler2", StandardScaler()),
            ]
        )

        numeric_a_pipeline = SklearnPipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("inner", inner_pipeline),
            ]
        )

        numeric_b_pipeline = SklearnPipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]
        )

        preprocessor = ColumnTransformer(
            [
                ("numeric_a", numeric_a_pipeline, numeric_features_a),
                ("numeric_b", numeric_b_pipeline, numeric_features_b),
            ]
        )

        expr = xo.memtable(data)
        schema = get_schema_out(preprocessor, expr, features=all_features)

        # All features should be in the output schema with sklearn-style prefixes
        # Pipeline doesn't add prefixes, so CT adds: numeric_a__col, numeric_b__col
        for feature in numeric_features_a:
            assert f"numeric_a__{feature}" in schema.names, (
                f"numeric_a__{feature} not in schema"
            )
        for feature in numeric_features_b:
            assert f"numeric_b__{feature}" in schema.names, (
                f"numeric_b__{feature} not in schema"
            )

        # Should have exactly the same number of columns
        assert len(schema.names) == len(all_features)

    def test_multiple_kv_encoded_columns(self):
        """Test that multiple KV-encoded transformers produce separate KV columns.

        Two OneHotEncoders should produce two separate KV-encoded columns,
        not pollute the entire output into one KV column.
        """
        import numpy as np
        import pandas as pd
        from sklearn.compose import ColumnTransformer
        from sklearn.preprocessing import OneHotEncoder, StandardScaler

        from xorq.expr.ml.structer import KV_ENCODED_TYPE, get_schema_out

        np.random.seed(42)
        n_samples = 20

        data = pd.DataFrame(
            {
                "age": np.random.randint(18, 80, n_samples).astype(float),
                "income": np.random.randint(20000, 150000, n_samples).astype(float),
                "education": np.random.choice(
                    ["high_school", "bachelor", "master"], n_samples
                ),
                "region": np.random.choice(["north", "south", "east"], n_samples),
            }
        )

        # Two separate OneHotEncoders for different categorical columns
        ct = ColumnTransformer(
            [
                ("num", StandardScaler(), ["age", "income"]),
                ("cat1", OneHotEncoder(sparse_output=False), ["education"]),
                ("cat2", OneHotEncoder(sparse_output=False), ["region"]),
            ]
        )

        expr = xo.memtable(data)
        schema = get_schema_out(ct, expr)

        # Known-schema numeric columns with prefixes
        assert "num__age" in schema.names
        assert "num__income" in schema.names

        # Each KV-encoded transformer gets its own column (no pollution!)
        assert "cat1" in schema.names
        assert "cat2" in schema.names
        assert schema["cat1"] == KV_ENCODED_TYPE
        assert schema["cat2"] == KV_ENCODED_TYPE

        # Should have exactly 4 columns: 2 numeric + 2 KV
        assert len(schema.names) == 4

    def test_hybrid_output_preserves_types(self):
        """Test that hybrid output preserves correct types for each column."""
        import pandas as pd
        from sklearn.compose import ColumnTransformer
        from sklearn.preprocessing import OneHotEncoder, StandardScaler

        import xorq.expr.datatypes as dt
        from xorq.expr.ml.structer import KV_ENCODED_TYPE, get_schema_out

        data = pd.DataFrame(
            {
                "num": [1.0, 2.0, 3.0],
                "cat": ["a", "b", "c"],
                "other": [4, 5, 6],  # int column
            }
        )

        ct = ColumnTransformer(
            [
                ("scaler", StandardScaler(), ["num"]),
                ("encoder", OneHotEncoder(sparse_output=False), ["cat"]),
                ("pass", "passthrough", ["other"]),
            ]
        )

        expr = xo.memtable(data)
        schema = get_schema_out(ct, expr)

        # Check types are preserved
        assert schema["scaler__num"] == dt.float64
        assert schema["encoder"] == KV_ENCODED_TYPE
        assert schema["pass__other"] == dt.int64
