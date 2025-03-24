use std::any::Any;
use std::fmt::Formatter;
use std::pin::Pin;
use std::sync::Arc;
use std::task::{Context, Poll};
use std::thread;

use arrow::array::RecordBatch;
use arrow::datatypes::SchemaRef;
use arrow::error::ArrowError;
use arrow::pyarrow::PyArrowType;
use datafusion::arrow::error::Result as ArrowResult;
use datafusion::execution::{SendableRecordBatchStream, TaskContext};
use datafusion::physical_plan::stream::RecordBatchStreamAdapter;
use datafusion::physical_plan::{DisplayAs, DisplayFormatType, ExecutionPlan, PlanProperties};
use datafusion_common::project_schema;
use futures::{Stream, TryStreamExt};
use pyo3::types::PyIterator;
use pyo3::{Bound, PyAny, PyObject, Python};

use crate::errors::DataFusionError;
use crate::utils::compute_properties;

use pyo3::prelude::*;

struct RecordBatchReaderAdapter {
    record_batch_reader: PyObject,
    columns: Option<Vec<String>>,
}

impl Stream for RecordBatchReaderAdapter {
    type Item = ArrowResult<RecordBatch>;

    fn poll_next(self: Pin<&mut Self>, _cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        thread::scope(|s| {
            let res = s
                .spawn(move || {
                    let option = Python::with_gil(|py| {
                        let batches = self.record_batch_reader.bind(py);
                        let mut batches = PyIterator::from_object(batches).unwrap();
                        Some(
                            batches
                                .next()?
                                .and_then(|batch| {
                                    let record_batch = batch
                                        .call_method1("select", (self.columns.clone().unwrap(),));
                                    let record_batch: RecordBatch =
                                        record_batch?.extract::<PyArrowType<_>>()?.0;
                                    Ok(record_batch)
                                })
                                .map_err(|err| ArrowError::ExternalError(Box::new(err))),
                        )
                    });

                    match option {
                        Some(Ok(value)) => Poll::Ready(Some(Ok(value))),
                        _ => Poll::Ready(None),
                    }
                })
                .join();

            match res {
                Ok(val) => val,
                _ => Poll::Ready(None),
            }
        })
    }
}

#[derive(Debug)]
pub struct IbisTableExec {
    record_batch_reader: PyObject,
    schema: SchemaRef,
    columns: Option<Vec<String>>,
    cache: PlanProperties,
}

impl IbisTableExec {
    pub(crate) fn new(
        _py: Python,
        record_batch_reader: &Bound<'_, PyAny>,
        projections: Option<&Vec<usize>>,
    ) -> Result<Self, DataFusionError> {
        // TODO use indices instead of columns
        let columns: Option<Result<Vec<String>, DataFusionError>> = projections.map(|p| {
            p.iter()
                .map(|index| {
                    let name: String = record_batch_reader
                        .getattr("schema")?
                        .call_method1("field", (*index,))?
                        .getattr("name")?
                        .extract()?;
                    Ok(name)
                })
                .collect()
        });
        let columns: Option<Vec<String>> = columns.transpose()?;

        let schema: SchemaRef = Arc::new(
            record_batch_reader
                .getattr("schema")?
                .extract::<PyArrowType<_>>()?
                .0,
        );
        let schema = project_schema(&schema, projections)?;

        let cache = compute_properties(schema.clone());

        Ok(IbisTableExec {
            record_batch_reader: record_batch_reader.clone().unbind(),
            schema,
            columns,
            cache,
        })
    }
}

impl DisplayAs for IbisTableExec {
    fn fmt_as(&self, _t: DisplayFormatType, f: &mut Formatter) -> std::fmt::Result {
        write!(f, "IbisTableExec")
    }
}

impl ExecutionPlan for IbisTableExec {
    fn name(&self) -> &str {
        "ibis_table"
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn schema(&self) -> SchemaRef {
        self.schema.clone()
    }

    fn properties(&self) -> &PlanProperties {
        &self.cache
    }

    fn children(&self) -> Vec<&Arc<dyn ExecutionPlan>> {
        // this is a leaf node and has no children
        vec![]
    }

    fn with_new_children(
        self: Arc<Self>,
        _children: Vec<Arc<dyn ExecutionPlan>>,
    ) -> datafusion_common::Result<Arc<dyn ExecutionPlan>> {
        Ok(self)
    }

    fn execute(
        &self,
        _partition: usize,
        _context: Arc<TaskContext>,
    ) -> datafusion_common::Result<SendableRecordBatchStream> {
        Python::with_gil(|py| {
            let record_batches = RecordBatchReaderAdapter {
                record_batch_reader: self.record_batch_reader.clone_ref(py),
                columns: self.columns.clone(),
            };

            let record_batch_stream: SendableRecordBatchStream =
                Box::pin(RecordBatchStreamAdapter::new(
                    self.schema.clone(),
                    record_batches.map_err(|e| e.into()),
                ));
            Ok(record_batch_stream)
        })
    }
}
