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
use pyo3::types::{IntoPyDict, PyTuple};
use pyo3::{pyclass, pymethods, PyAny, PyObject, PyResult, Python};

use crate::ibis_filter_expression::IbisFilterExpression;
use crate::ibis_table_exec::IbisTableExec;

#[pyclass(name = "TableProvider", module = "datafusion", subclass)]
#[derive(Debug, Clone)]
pub struct PyTableProvider {
    table_provider: PyObject,
}

#[pymethods]
impl PyTableProvider {
    #[new]
    #[pyo3(signature=(table_provider))]
    pub fn new(table_provider: &PyAny) -> PyResult<Self> {
        Ok(Self {
            table_provider: table_provider.into(),
        })
    }
}

#[async_trait]
impl TableProvider for PyTableProvider {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn schema(&self) -> SchemaRef {
        Python::with_gil(|py| {
            let table_provider = self.table_provider.as_ref(py);
            Arc::new(
                table_provider
                    .call_method0("schema")
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
            let kwargs = [("filters", ibis_filters)].into_py_dict(py);

            let table = self
                .table_provider
                .call_method(py, "scan", (), Some(kwargs))
                .unwrap();

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
            .map(|&filter| match IbisFilterExpression::try_from(filter) {
                Ok(_) => Ok(TableProviderFilterPushDown::Exact),
                _ => Ok(TableProviderFilterPushDown::Unsupported),
            })
            .collect()
    }
}
