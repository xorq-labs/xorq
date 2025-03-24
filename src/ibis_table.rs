use std::any::Any;
use std::sync::Arc;

use arrow::datatypes::SchemaRef;
use arrow::pyarrow::PyArrowType;
use async_trait::async_trait;
use datafusion::catalog::Session;
use datafusion::datasource::TableProvider;
use datafusion::physical_plan::ExecutionPlan;
use datafusion_common::DataFusionError;
use datafusion_expr::{Expr, TableProviderFilterPushDown, TableType};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::PyModule;
use pyo3::types::{PyTuple, PyType};
use pyo3::{PyAny, PyObject, PyResult, Python};

use crate::ibis_filter_expression::IbisFilterExpression;
use crate::ibis_table_exec::IbisTableExec;

use crate::errors::to_datafusion_err;
use pyo3::prelude::*;

// Wraps an ibis.Table class and implements a Datafusion TableProvider around it
#[derive(Debug)]
pub(crate) struct IbisTable {
    ibis_table: PyObject,
}

impl IbisTable {
    // Creates a Python ibis.Table
    pub fn new(ibis_table: &Bound<'_, PyAny>, py: Python) -> PyResult<Self> {
        let pa = PyModule::import(py, "ibis.expr.types")?;
        let table = pa.getattr("Table")?;
        let table_type = table.downcast::<PyType>()?;

        if ibis_table.is_instance(table_type)? {
            Ok(IbisTable {
                ibis_table: ibis_table.clone().unbind(),
            })
        } else {
            Err(PyValueError::new_err(
                "ibis_table argument must be a ibis.expr.types.Table object",
            ))
        }
    }
}

#[async_trait]
impl TableProvider for IbisTable {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn schema(&self) -> SchemaRef {
        Python::with_gil(|py| {
            let batch_reader = self.ibis_table.bind(py);
            Arc::new(
                batch_reader
                    .call_method0("schema")
                    .unwrap()
                    .call_method0("to_pyarrow")
                    .unwrap()
                    .extract::<PyArrowType<_>>()
                    .unwrap()
                    .0,
            )
        })
    }

    fn table_type(&self) -> TableType {
        TableType::Base
    }

    async fn scan(
        &self,
        _state: &dyn Session,
        projection: Option<&Vec<usize>>,
        filters: &[Expr],
        _limit: Option<usize>,
    ) -> datafusion_common::Result<Arc<dyn ExecutionPlan>> {
        Python::with_gil(|py| {
            let table = if !filters.is_empty() {
                let args = filters
                    .iter()
                    .map(|filter| {
                        IbisFilterExpression::try_from(filter)
                            .unwrap()
                            .inner()
                            .clone_ref(py)
                    })
                    .collect::<Vec<PyObject>>();
                let ibis_filters = PyTuple::new(py, &args).map_err(to_datafusion_err)?;
                self.ibis_table
                    .call_method1(py, "filter", ibis_filters)
                    .map_err(|err| DataFusionError::Execution(format!("{err}")))?
                    .call_method0(py, "to_pyarrow_batches")
                    .unwrap()
            } else {
                self.ibis_table
                    .call_method0(py, "to_pyarrow_batches")
                    .unwrap()
            };

            let plan: Arc<dyn ExecutionPlan> = Arc::new(
                IbisTableExec::new(py, table.bind(py), projection)
                    .map_err(|err| DataFusionError::External(Box::new(err)))?,
            );
            Ok(plan)
        })
    }

    fn supports_filters_pushdown(
        &self,
        filters: &[&Expr],
    ) -> datafusion_common::Result<Vec<TableProviderFilterPushDown>> {
        filters
            .iter()
            .map(|&f| match IbisFilterExpression::try_from(f) {
                Ok(_) => Ok(TableProviderFilterPushDown::Exact),
                _ => Ok(TableProviderFilterPushDown::Unsupported),
            })
            .collect()
    }
}
