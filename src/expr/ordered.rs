use datafusion_expr::{Expr, SortExpr};
use pyo3::{pyclass, pymethods};
use std::clone::Clone;

use crate::expr::PyExpr;

#[pyclass(name = "Ordered", module = "datafusion.expr", subclass)]
#[derive(Clone, Debug)]
pub struct PyOrdered {
    pub expr: PyExpr,
    pub asc: bool,
    pub nulls_first: bool,
}

impl From<SortExpr> for PyOrdered {
    fn from(value: SortExpr) -> PyOrdered {
        PyOrdered {
            expr: PyExpr::from(value.expr.clone()),
            asc: value.asc,
            nulls_first: value.nulls_first,
        }
    }
}

impl From<PyOrdered> for SortExpr {
    fn from(value: PyOrdered) -> SortExpr {
        SortExpr {
            expr: Expr::from(value.expr),
            asc: value.asc,
            nulls_first: value.nulls_first,
        }
    }
}

#[pymethods]
impl PyOrdered {
    #[new]
    pub(crate) fn new(expr: PyExpr, asc: bool, nulls_first: bool) -> Self {
        Self {
            expr,
            asc,
            nulls_first,
        }
    }

    fn expr(&self) -> PyExpr {
        self.expr.clone()
    }

    fn asc(&self) -> bool {
        self.asc
    }

    fn nulls_first(&self) -> bool {
        self.nulls_first
    }
}
