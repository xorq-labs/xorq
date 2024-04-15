use datafusion::arrow::datatypes::{Schema, SchemaRef};
use datafusion::arrow::pyarrow::PyArrowType;
use datafusion_expr::builder::table_scan_with_filters;
use datafusion_expr::expr::Sort;
use datafusion_expr::{table_scan, Expr, LogicalPlanBuilder};
use pyo3::prelude::PyModule;
use pyo3::{pyclass, pyfunction, pymethods, wrap_pyfunction, PyResult};

use crate::expr::ordered::PyOrdered;
use crate::expr::PyExpr;
use crate::sql::logical::PyLogicalPlan;

#[pyclass(name = "LogicalPlanBuilder", module = "datafusion", subclass)]
#[derive(Clone)]
pub struct PyLogicalPlanBuilder {
    builder: LogicalPlanBuilder,
}

impl From<LogicalPlanBuilder> for PyLogicalPlanBuilder {
    fn from(builder: LogicalPlanBuilder) -> Self {
        Self { builder }
    }
}

#[pymethods]
impl PyLogicalPlanBuilder {
    pub fn build(&self) -> PyLogicalPlan {
        PyLogicalPlan::from(self.builder.clone().build().unwrap())
    }
    pub fn filter(&self, expr: PyExpr) -> Self {
        let expr = expr.expr;

        Self {
            builder: self.builder.clone().filter(expr).unwrap(),
        }
    }

    pub fn project(&self, expr: Vec<PyExpr>) -> Self {
        let expr = expr.iter().map(|e| Expr::from(e.clone()));
        Self {
            builder: self.builder.clone().project(expr).unwrap(),
        }
    }

    pub fn sort(&self, expr: Vec<PyOrdered>) -> Self {
        let expr = expr.iter().map(|e| Expr::Sort(Sort::from(e.clone())));
        Self {
            builder: self.builder.clone().sort(expr).unwrap(),
        }
    }
}

#[pyfunction]
#[pyo3(name = "table_scan")]
pub fn py_table_scan(
    name: &str,
    table_schema: PyArrowType<Schema>,
    projections: Option<Vec<usize>>,
) -> PyLogicalPlanBuilder {
    let plan_builder =
        table_scan(Some(name), &SchemaRef::from(table_schema.0), projections).unwrap();

    PyLogicalPlanBuilder::from(plan_builder)
}

#[pyfunction]
#[pyo3(name = "table_scan_with_filters")]
pub fn py_table_scan_with_filters(
    name: &str,
    table_schema: PyArrowType<Schema>,
    filters: Vec<PyExpr>,
    projections: Option<Vec<usize>>,
) -> PyLogicalPlanBuilder {
    let filters_exprs = filters.iter().map(|e| Expr::from(e.clone())).collect();

    let plan_builder = table_scan_with_filters(
        Some(name),
        &SchemaRef::from(table_schema.0),
        projections,
        filters_exprs,
    )
    .unwrap();
    PyLogicalPlanBuilder::from(plan_builder)
}

pub(crate) fn init_module(m: &PyModule) -> PyResult<()> {
    m.add_wrapped(wrap_pyfunction!(py_table_scan))?;
    m.add_wrapped(wrap_pyfunction!(py_table_scan_with_filters))?;
    Ok(())
}
