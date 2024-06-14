use std::any::Any;
use std::sync::Arc;

use arrow::datatypes::SchemaRef;
use arrow::pyarrow::PyArrowType;
use async_trait::async_trait;
use datafusion::datasource::TableProvider;
use datafusion::execution::context::SessionState;
use datafusion::physical_plan::ExecutionPlan;
use datafusion_common::DataFusionError;
use datafusion_expr::{Expr, TableProviderFilterPushDown, TableType};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::PyModule;
use pyo3::types::{PyTuple, PyType};
use pyo3::{PyAny, PyObject, PyResult, Python};

use crate::ibis_filter_expression::IbisFilterExpression;
use crate::ibis_table_exec::IbisTableExec;

// Wraps an ibis.Table class and implements a Datafusion TableProvider around it
#[derive(Debug, Clone)]
pub(crate) struct IbisTable {
    ibis_table: PyObject,
}

impl IbisTable {
    // Creates a Python ibis.Table
    pub fn new(ibis_table: &PyAny, py: Python) -> PyResult<Self> {
        let pa = PyModule::import(py, "ibis.expr.types")?;
        let table: &PyType = pa.getattr("Table")?.downcast()?;
        if ibis_table.is_instance(table)? {
            Ok(IbisTable {
                ibis_table: ibis_table.into(),
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
            let batch_reader = self.ibis_table.as_ref(py);
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
        _state: &SessionState,
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
                            .clone()
                    })
                    .collect::<Vec<PyObject>>();
                let ibis_filters = PyTuple::new(py, &args);
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
                IbisTableExec::new(py, table.as_ref(py), projection)
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
