from __future__ import annotations

from xorq.vendor.ibis.formats.pandas import PandasData


class DuckDBPandasData(PandasData):
    @staticmethod
    def convert_Array(s, dtype, pandas_type):
        return s.replace(float("nan"), None)


try:
    from xorq.vendor.ibis.formats.pyarrow import PyArrowData

    class DuckDBPyArrowData(PyArrowData):
        @staticmethod
        def convert_scalar(scalar, dtype):
            import pyarrow as pa

            result = PyArrowData.convert_scalar(scalar, dtype)
            if isinstance(result, pa.Scalar) and result.as_py() is None:
                return pa.scalar(None, type=result.type)
            return result

        @staticmethod
        def convert_column(column, dtype):
            import pyarrow as pa

            result = PyArrowData.convert_column(column, dtype)
            if isinstance(result, pa.ChunkedArray):
                if result.null_count == len(result):
                    return pa.chunked_array(
                        [pa.array([None] * len(result), type=result.type)]
                    )
            return result

except ImportError:
    pass
