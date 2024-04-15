// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements.  See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership.  The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License.  You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied.  See the License for the
// specific language governing permissions and limitations
// under the License.

use std::any::Any;
use std::ops::DerefMut;
use std::pin::Pin;
use std::sync::{Arc, Mutex};
use std::{fmt, thread};

use futures::stream::Stream;
use futures::task::{Context, Poll};

use datafusion::arrow::datatypes::SchemaRef;
use datafusion::arrow::ffi_stream::ArrowArrayStreamReader;
use datafusion::arrow::record_batch::{RecordBatch, RecordBatchReader};
use datafusion::datasource::{TableProvider, TableType};
use datafusion::error::Result;
use datafusion::execution::context::SessionState;
use datafusion::execution::TaskContext;
use datafusion::physical_plan::expressions::PhysicalSortExpr;
use datafusion::physical_plan::stream::RecordBatchStreamAdapter;
use datafusion::physical_plan::{
    project_schema, DisplayAs, DisplayFormatType, ExecutionPlan, SendableRecordBatchStream,
};
use datafusion_common::DataFusionError;
use datafusion_expr::Expr;

use async_trait::async_trait;

#[derive(Clone, Debug)]
pub struct PyRecordBatchProvider {
    reader: Arc<Mutex<Option<ArrowArrayStreamReader>>>,
    schema: SchemaRef,
}

impl PyRecordBatchProvider {
    pub(crate) async fn create_physical_plan(
        &self,
        projections: Option<&Vec<usize>>,
        schema: SchemaRef,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        Ok(Arc::new(CustomExec::new(self.clone(), projections, schema)))
    }

    pub fn new(aasr: ArrowArrayStreamReader) -> Self {
        let schema = aasr.schema();
        Self {
            reader: Arc::new(Mutex::new(aasr.into())),
            schema,
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
        _state: &SessionState,
        projection: Option<&Vec<usize>>,
        _filter: &[Expr],
        _limit: Option<usize>,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        return self.create_physical_plan(projection, self.schema()).await;
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
struct CustomExec {
    record_batch_provider: PyRecordBatchProvider,
    projected_schema: SchemaRef,
}

impl CustomExec {
    fn new(
        record_batch_provider: PyRecordBatchProvider,
        projections: Option<&Vec<usize>>,
        schema: SchemaRef,
    ) -> Self {
        let projected_schema = project_schema(&schema, projections).unwrap();
        Self {
            record_batch_provider,
            projected_schema,
        }
    }
}

impl DisplayAs for CustomExec {
    fn fmt_as(&self, _t: DisplayFormatType, f: &mut fmt::Formatter) -> std::fmt::Result {
        write!(f, "CustomExec")
    }
}

impl ExecutionPlan for CustomExec {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn schema(&self) -> SchemaRef {
        self.projected_schema.clone()
    }

    fn output_partitioning(&self) -> datafusion::physical_plan::Partitioning {
        datafusion::physical_plan::Partitioning::UnknownPartitioning(1)
    }

    fn output_ordering(&self) -> Option<&[PhysicalSortExpr]> {
        None
    }

    fn children(&self) -> Vec<Arc<dyn ExecutionPlan>> {
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
        let record_batch_stream: SendableRecordBatchStream = Box::pin(
            RecordBatchStreamAdapter::new(self.schema(), record_batch_provider),
        );
        Ok(record_batch_stream)
    }
}
