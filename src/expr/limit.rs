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

use crate::common::df_schema::PyDFSchema;
use crate::errors::to_datafusion_err;
use crate::expr::logical_node::LogicalNode;
use crate::sql::logical::PyLogicalPlan;
use datafusion_common::ScalarValue;
use datafusion_expr::logical_plan::Limit;
use datafusion_expr::Expr;
use pyo3::prelude::*;
use std::fmt::{self, Display, Formatter};

#[pyclass(name = "Limit", module = "datafusion.expr", subclass)]
#[derive(Clone)]
pub struct PyLimit {
    limit: Limit,
}

impl From<Limit> for PyLimit {
    fn from(limit: Limit) -> PyLimit {
        PyLimit { limit }
    }
}

impl From<PyLimit> for Limit {
    fn from(limit: PyLimit) -> Self {
        limit.limit
    }
}

impl Display for PyLimit {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        write!(
            f,
            "Limit
            Skip: {:?}
            Fetch: {:?}
            Input: {:?}",
            &self.limit.skip, &self.limit.fetch, &self.limit.input
        )
    }
}

#[pymethods]
impl PyLimit {
    /// Retrieves the skip value for this `Limit`
    fn skip(&self) -> usize {
        match self.limit.skip.as_deref() {
            Some(expr) => match *expr {
                Expr::Literal(ScalarValue::Int64(s)) => {
                    // `skip = NULL` is equivalent to `skip = 0`
                    let s = s.unwrap_or(0);
                    if s >= 0 {
                        s as usize
                    } else {
                        panic!("OFFSET must be >=0, '{}' was provided", s)
                    }
                }
                _ => panic!("Unsupported Expr for OFFSET"),
            },
            // `skip = None` is equivalent to `skip = 0`
            None => 0,
        }
    }

    /// Retrieves the fetch value for this `Limit`
    fn fetch(&self) -> Option<usize> {
        match self.limit.fetch.as_deref() {
            Some(expr) => match *expr {
                Expr::Literal(ScalarValue::Int64(Some(s))) => {
                    if s >= 0 {
                        Some(s as usize)
                    } else {
                        None
                    }
                }
                Expr::Literal(ScalarValue::Int64(None)) => None,
                _ => None,
            },
            None => None,
        }
    }

    /// Retrieves the input `LogicalPlan` to this `Limit` node
    fn input(&self) -> PyResult<Vec<PyLogicalPlan>> {
        Ok(Self::inputs(self))
    }

    /// Resulting Schema for this `Limit` node instance
    fn schema(&self) -> PyResult<PyDFSchema> {
        Ok(self.limit.input.schema().as_ref().clone().into())
    }

    fn __repr__(&self) -> PyResult<String> {
        Ok(format!("Limit({})", self))
    }
}

impl LogicalNode for PyLimit {
    fn inputs(&self) -> Vec<PyLogicalPlan> {
        vec![PyLogicalPlan::from((*self.limit.input).clone())]
    }

    fn to_variant(&self, py: Python) -> PyResult<PyObject> {
        Ok(self
            .clone()
            .into_pyobject(py)
            .map_err(to_datafusion_err)?
            .into_any()
            .unbind())
    }
}
