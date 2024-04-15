use datafusion_expr::expr::Sort;
use datafusion_expr::Expr;
use pyo3::{pyclass, pymethods};

use crate::expr::PyExpr;

#[pyclass(name = "Ordered", module = "datafusion.expr", subclass)]
#[derive(Clone, Debug)]
pub struct PyOrdered {
    pub expr: PyExpr,
    pub asc: bool,
    pub nulls_first: bool,
}

impl From<Sort> for PyOrdered {
    fn from(sort: Sort) -> PyOrdered {
        PyOrdered {
            expr: PyExpr::from(*sort.expr.clone()),
            asc: sort.asc,
            nulls_first: sort.nulls_first,
        }
    }
}

impl From<PyOrdered> for Sort {
    fn from(value: PyOrdered) -> Sort {
        Sort {
            expr: Box::from(Expr::from(value.expr)),
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
