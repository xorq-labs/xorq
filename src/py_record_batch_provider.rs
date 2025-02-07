use std::any::Any;
use std::ops::DerefMut;
use std::pin::Pin;
use std::sync::{Arc, Mutex};
use std::{fmt, thread};

use crate::utils::compute_properties_with_orderings;
use async_trait::async_trait;
use datafusion::arrow::datatypes::SchemaRef;
use datafusion::arrow::ffi_stream::ArrowArrayStreamReader;
use datafusion::arrow::record_batch::{RecordBatch, RecordBatchReader};
use datafusion::catalog::Session;
use datafusion::datasource::{TableProvider, TableType};
use datafusion::error::Result;
use datafusion::execution::TaskContext;
use datafusion::physical_expr::LexOrdering;
use datafusion::physical_plan::stream::RecordBatchStreamAdapter;
use datafusion::physical_plan::{
    project_schema, DisplayAs, DisplayFormatType, ExecutionPlan, PlanProperties,
    SendableRecordBatchStream,
};
use datafusion_common::DataFusionError;
use datafusion_expr::Expr;
use futures::stream::Stream;
use futures::task::{Context, Poll};

#[derive(Clone, Debug)]
pub struct PyRecordBatchProvider {
    reader: Arc<Mutex<Option<ArrowArrayStreamReader>>>,
    schema: SchemaRef,
    ordering: LexOrdering,
}

impl PyRecordBatchProvider {
    pub(crate) async fn create_physical_plan(
        &self,
        projections: Option<&Vec<usize>>,
        schema: SchemaRef,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        Ok(Arc::new(PyRecordBatchProviderExec::new(
            self.clone(),
            projections,
            schema,
        )))
    }

    pub fn new(aasr: ArrowArrayStreamReader, ordering: LexOrdering) -> Self {
        let schema = aasr.schema();
        Self {
            reader: Arc::new(Mutex::new(aasr.into())),
            schema,
            ordering,
        }
    }
}

#[async_trait]
impl TableProvider for PyRecordBatchProvider {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn schema(&self) -> SchemaRef {
        self.schema.clone()
    }

    fn table_type(&self) -> TableType {
        TableType::Base
    }

    async fn scan(
        &self,
        _state: &dyn Session,
        projection: Option<&Vec<usize>>,
        _filter: &[Expr],
        _limit: Option<usize>,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        self.create_physical_plan(projection, self.schema()).await
    }
}

impl Stream for PyRecordBatchProvider {
    type Item = Result<RecordBatch>;

    fn poll_next(self: Pin<&mut Self>, _cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        match self.reader.lock().unwrap().deref_mut() {
            Some(ref mut reader) => thread::scope(|s| {
                let res = s
                    .spawn(move || match reader.next() {
                        Some(value) => Poll::Ready(Some(value)).map_err(DataFusionError::from),
                        None => Poll::Ready(None),
                    })
                    .join();

                match res {
                    Ok(val) => val,
                    _ => Poll::Ready(None),
                }
            }),
            _ => Poll::Ready(None),
        }
    }
}

#[derive(Debug, Clone)]
struct ProjectedPyRecordBatchProvider {
    record_batch_provider: PyRecordBatchProvider,
    projections: Vec<usize>,
}

impl ProjectedPyRecordBatchProvider {
    fn new(record_batch_provider: PyRecordBatchProvider, projections: Vec<usize>) -> Self {
        let projections = projections.clone();
        Self {
            record_batch_provider,
            projections,
        }
    }
}

impl Stream for ProjectedPyRecordBatchProvider {
    type Item = Result<RecordBatch>;

    fn poll_next(self: Pin<&mut Self>, _cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        let projections = self.projections.clone();
        match self
            .record_batch_provider
            .reader
            .lock()
            .unwrap()
            .deref_mut()
        {
            Some(ref mut reader) => thread::scope(|s| {
                let res = s
                    .spawn(move || match reader.next() {
                        Some(value) => Poll::Ready(Some(
                            value.map(|rb| rb.project(projections.as_slice()).unwrap()),
                        ))
                        .map_err(DataFusionError::from),
                        None => Poll::Ready(None),
                    })
                    .join();

                match res {
                    Ok(val) => val,
                    _ => Poll::Ready(None),
                }
            }),
            _ => Poll::Ready(None),
        }
    }
}

#[derive(Debug, Clone)]
struct PyRecordBatchProviderExec {
    record_batch_provider: PyRecordBatchProvider,
    projected_schema: SchemaRef,
    projections: Option<Vec<usize>>,
    plan_properties: PlanProperties,
}

impl PyRecordBatchProviderExec {
    fn new(
        record_batch_provider: PyRecordBatchProvider,
        projections: Option<&Vec<usize>>,
        schema: SchemaRef,
    ) -> Self {
        let projected_schema = project_schema(&schema, projections).unwrap();
        let projections = projections.map(|v| (*v).clone());
        let plan_properties = compute_properties_with_orderings(
            projected_schema.clone(),
            &[record_batch_provider.ordering.clone()],
        );
        Self {
            record_batch_provider,
            projected_schema,
            projections,
            plan_properties,
        }
    }
}

impl DisplayAs for PyRecordBatchProviderExec {
    fn fmt_as(&self, _t: DisplayFormatType, f: &mut fmt::Formatter) -> std::fmt::Result {
        if let Some(output_ordering) = self.plan_properties.output_ordering() {
            write!(
                f,
                "PyRecordBatchProviderExec ordering=[{}]",
                output_ordering
            )
        } else {
            write!(f, "PyRecordBatchProviderExec ordering=[None]")
        }
    }
}

impl ExecutionPlan for PyRecordBatchProviderExec {
    fn name(&self) -> &str {
        "py_record_batch_provider"
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn schema(&self) -> SchemaRef {
        self.projected_schema.clone()
    }

    fn properties(&self) -> &PlanProperties {
        &self.plan_properties
    }

    fn maintains_input_order(&self) -> Vec<bool> {
        // Tell optimizer this operator doesn't reorder its input
        vec![true]
    }

    fn children(&self) -> Vec<&Arc<dyn ExecutionPlan>> {
        vec![]
    }

    fn with_new_children(
        self: Arc<Self>,
        _: Vec<Arc<dyn ExecutionPlan>>,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        Ok(self)
    }

    fn execute(
        &self,
        _partition: usize,
        _context: Arc<TaskContext>,
    ) -> Result<SendableRecordBatchStream> {
        let record_batch_provider = self.record_batch_provider.clone();
        let projections = self.projections.clone();
        let projected_schema = self.projected_schema.clone();

        let record_batch_stream: SendableRecordBatchStream = if let Some(pj) = projections {
            let record_batch_provider =
                ProjectedPyRecordBatchProvider::new(record_batch_provider.clone(), pj.clone());
            Box::pin(RecordBatchStreamAdapter::new(
                projected_schema.clone(),
                record_batch_provider,
            ))
        } else {
            Box::pin(RecordBatchStreamAdapter::new(
                projected_schema.clone(),
                record_batch_provider,
            ))
        };
        Ok(record_batch_stream)
    }
}
