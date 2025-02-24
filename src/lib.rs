use crate::sql::{builder, parser};
use pyo3::prelude::*;
#[allow(clippy::borrow_deref_ref)]
pub mod catalog;
pub mod common;
#[allow(clippy::borrow_deref_ref)]
mod context;
mod dataframe;
mod dataset;
mod dataset_exec;
mod errors;
pub mod expr;
mod functions;
mod ibis_filter_expression;
mod ibis_table;
mod ibis_table_exec;
pub mod model;
mod optimizer;
pub mod physical_plan;
mod provider;
mod py_record_batch_provider;
mod pyarrow_filter_expression;
mod record_batch;
pub mod sql;
#[allow(clippy::borrow_deref_ref)]
mod udaf;
#[allow(clippy::borrow_deref_ref)]
mod udf;
mod udwf;
pub mod utils;

mod object_storage;

// Used to define Tokio Runtime as a Python module attribute
#[pyclass]
pub(crate) struct TokioRuntime(tokio::runtime::Runtime);

/// Low-level LetSQL internal package.
#[pymodule]
fn _internal(py: Python, m: Bound<'_, PyModule>) -> PyResult<()> {
    // Register the Tokio Runtime as a module attribute, so we can reuse it
    m.add(
        "runtime",
        TokioRuntime(tokio::runtime::Runtime::new().unwrap()),
    )?;

    m.add_class::<context::PySessionConfig>()?;
    m.add_class::<context::PySessionContext>()?;
    m.add_class::<context::PySessionState>()?;
    m.add_class::<dataframe::PyDataFrame>()?;
    m.add_class::<udf::PyScalarUDF>()?;
    m.add_class::<udaf::PyAggregateUDF>()?;
    m.add_class::<udwf::PyWindowUDF>()?;
    m.add_class::<sql::logical::PyLogicalPlan>()?;
    m.add_class::<physical_plan::PyExecutionPlan>()?;
    m.add_class::<parser::PyContextProvider>()?;
    m.add_class::<builder::PyLogicalPlanBuilder>()?;
    m.add_class::<optimizer::PyOptimizerContext>()?;
    m.add_class::<optimizer::PyOptimizerRule>()?;
    m.add_class::<provider::PyTableProvider>()?;
    m.add_class::<catalog::PyTable>()?;

    // Register `common` as a submodule. Matching `datafusion-common` https://docs.rs/datafusion-common/latest/datafusion_common/
    let common = PyModule::new_bound(py, "common")?;
    common::init_module(&common)?;
    m.add_submodule(&common)?;

    // Register `expr` as a submodule. Matching `datafusion-expr` https://docs.rs/datafusion-expr/latest/datafusion_expr/
    let expr = PyModule::new_bound(py, "expr")?;
    expr::init_module(&expr)?;
    m.add_submodule(&expr)?;

    let parser = PyModule::new_bound(py, "parser")?;
    parser::init_module(&parser)?;
    m.add_submodule(&parser)?;

    let optimizer = PyModule::new_bound(py, "optimizer")?;
    optimizer::init_module(&optimizer)?;
    m.add_submodule(&optimizer)?;

    let builder = PyModule::new_bound(py, "builder")?;
    builder::init_module(&builder)?;
    m.add_submodule(&builder)?;

    Ok(())
}
