use pyo3::prelude::*;
#[allow(clippy::borrow_deref_ref)]
pub mod catalog;
#[allow(clippy::borrow_deref_ref)]
mod context;
mod dataframe;
mod dataset;
mod dataset_exec;
mod errors;
mod pyarrow_filter_expression;
#[allow(clippy::borrow_deref_ref)]
mod udaf;
#[allow(clippy::borrow_deref_ref)]
mod udf;
pub mod utils;

pub mod predict_udf;

// Used to define Tokio Runtime as a Python module attribute
#[pyclass]
pub(crate) struct TokioRuntime(tokio::runtime::Runtime);

/// Low-level LetSQL internal package.
#[pymodule]
fn _internal(_py: Python, m: &PyModule) -> PyResult<()> {
    // Register the Tokio Runtime as a module attribute so we can reuse it
    m.add(
        "runtime",
        TokioRuntime(tokio::runtime::Runtime::new().unwrap()),
    )?;

    m.add_class::<context::PySessionConfig>()?;
    m.add_class::<context::PySessionContext>()?;
    m.add_class::<dataframe::PyDataFrame>()?;
    m.add_class::<udf::PyScalarUDF>()?;
    m.add_class::<udaf::PyAggregateUDF>()?;
    Ok(())
}
