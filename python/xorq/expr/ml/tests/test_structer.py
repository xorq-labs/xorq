import pandas as pd
import pytest

import xorq.api as xo
import xorq.expr.datatypes as dt
from xorq.expr.ml.structer import (
    KV_ENCODED_TYPE,
    KVEncoder,
    Structer,
)


sklearn = pytest.importorskip("sklearn")


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
        """Test KVEncoder.decode expands to individual columns."""
        df = pd.DataFrame(
            {
                "encoded": [
                    ({"key": "a", "value": 1.0}, {"key": "b", "value": 0.0}),
                    ({"key": "a", "value": 0.0}, {"key": "b", "value": 1.0}),
                ]
            }
        )

        result = KVEncoder.decode(df, "encoded")

        assert "a" in result.columns
        assert "b" in result.columns
        assert "encoded" not in result.columns
        assert result["a"].tolist() == [1.0, 0.0]
        assert result["b"].tolist() == [0.0, 1.0]

    def test_decode_preserves_other_columns(self):
        """Test that decode preserves non-encoded columns."""
        df = pd.DataFrame(
            {
                "id": [1, 2],
                "encoded": [
                    ({"key": "x", "value": 1.0},),
                    ({"key": "x", "value": 2.0},),
                ],
            }
        )

        result = KVEncoder.decode(df, "encoded")

        assert "id" in result.columns
        assert result["id"].tolist() == [1, 2]

    def test_decode_empty_dataframe(self):
        """Test decode handles empty DataFrame."""
        df = pd.DataFrame({"encoded": []})
        result = KVEncoder.decode(df, "encoded")
        assert "encoded" not in result.columns

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
        decoded = KVEncoder.decode(df, "transformed")
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
        decoded = KVEncoder.decode(df, "transformed")
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
        xorq_decoded = KVEncoder.decode(df, "transformed")

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


class TestMaybeDecodeEncodedColumns:
    """Tests for _maybe_decode_encoded_columns helper."""

    def test_no_encoded_cols(self):
        """Test passthrough when no encoded columns."""
        from xorq.expr.ml.fit_lib import _maybe_decode_encoded_columns

        df = pd.DataFrame({"a": [1.0, 2.0], "b": [3.0, 4.0]})
        features = ("a", "b")
        encoded_cols = ()

        result_df, result_features = _maybe_decode_encoded_columns(
            df, features, encoded_cols
        )

        pd.testing.assert_frame_equal(result_df, df)
        assert result_features == features

    def test_decode_encoded_column(self):
        """Test decoding KV-encoded column."""
        from xorq.expr.ml.fit_lib import _maybe_decode_encoded_columns

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

        result_df, result_features = _maybe_decode_encoded_columns(
            df, features, encoded_cols
        )

        assert "encoded" not in result_df.columns
        assert "x" in result_df.columns
        assert "y" in result_df.columns
        assert "other" in result_df.columns
        assert set(result_features) == {"x", "y", "other"}

    def test_missing_encoded_col_skipped(self):
        """Test that missing encoded columns are skipped."""
        from xorq.expr.ml.fit_lib import _maybe_decode_encoded_columns

        df = pd.DataFrame({"a": [1.0, 2.0]})
        features = ("a",)
        encoded_cols = ("nonexistent",)

        result_df, result_features = _maybe_decode_encoded_columns(
            df, features, encoded_cols
        )

        pd.testing.assert_frame_equal(result_df, df)
        assert result_features == features
