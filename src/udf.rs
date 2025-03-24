use pyo3::{prelude::*, types::PyTuple};

use crate::errors::to_datafusion_err;
use crate::utils::{make_scalar_function, parse_volatility};
use datafusion::arrow::array::{make_array, Array, ArrayData, ArrayRef};
use datafusion::arrow::datatypes::DataType;
use datafusion::arrow::pyarrow::{FromPyArrow, PyArrowType, ToPyArrow};
use datafusion::error::DataFusionError;
use datafusion_expr::create_udf;
use datafusion_expr::function::ScalarFunctionImplementation;
use datafusion_expr::ScalarUDF;

/// Create a DataFusion's UDF implementation from a python function
/// that expects pyarrow arrays. This is more efficient as it performs
/// a zero-copy of the contents.
fn to_rust_function(func: PyObject) -> ScalarFunctionImplementation {
    make_scalar_function(
        move |args: &[ArrayRef]| -> Result<ArrayRef, DataFusionError> {
            Python::with_gil(|py| {
                // 1. cast args to Pyarrow arrays
                let py_args = args
                    .iter()
                    .map(|arg| arg.into_data().to_pyarrow(py).unwrap())
                    .collect::<Vec<_>>();
                let py_args = PyTuple::new(py, py_args).map_err(to_datafusion_err)?;

                // 2. call function
                let value = func.call(py, py_args, None).map_err(to_datafusion_err)?;

                // 3. cast to arrow::array::Array
                let array_data = ArrayData::from_pyarrow_bound(value.bind(py)).unwrap();
                Ok(make_array(array_data))
            })
        },
    )
}

/// Represents a PyScalarUDF
#[pyclass(name = "ScalarUDF", module = "let", subclass)]
#[derive(Debug, Clone)]
pub struct PyScalarUDF {
    pub(crate) function: ScalarUDF,
}

#[pymethods]
impl PyScalarUDF {
    #[new]
    fn new(
        name: &str,
        func: PyObject,
        input_types: PyArrowType<Vec<DataType>>,
        return_type: PyArrowType<DataType>,
        volatility: &str,
    ) -> PyResult<Self> {
        let function = create_udf(
            name,
            input_types.0,
            return_type.0,
            parse_volatility(volatility)?,
            to_rust_function(func),
        );
        Ok(Self { function })
    }

    fn __repr__(&self) -> PyResult<String> {
        Ok(format!("ScalarUDF({})", self.function.name()))
    }
}
