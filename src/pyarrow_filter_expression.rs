use pyo3::prelude::*;

use std::convert::TryFrom;
use std::result::Result;

use datafusion_common::{Column, ScalarValue};
use datafusion_expr::{expr::InList, Between, BinaryExpr, Expr, Operator};

use crate::errors::{to_datafusion_err, DataFusionError};

#[derive(Debug)]
#[repr(transparent)]
pub(crate) struct PyArrowFilterExpression(PyObject);

fn operator_to_py<'py>(
    operator: &Operator,
    op: &Bound<'py, PyModule>,
) -> Result<Bound<'py, PyAny>, DataFusionError> {
    let py_op: Bound<'_, PyAny> = match operator {
        Operator::Eq => op.getattr("eq")?,
        Operator::NotEq => op.getattr("ne")?,
        Operator::Lt => op.getattr("lt")?,
        Operator::LtEq => op.getattr("le")?,
        Operator::Gt => op.getattr("gt")?,
        Operator::GtEq => op.getattr("ge")?,
        Operator::And => op.getattr("and_")?,
        Operator::Or => op.getattr("or_")?,
        _ => {
            return Err(DataFusionError::Common(format!(
                "Unsupported operator {operator:?}"
            )))
        }
    };
    Ok(py_op)
}

fn extract_scalar_list(exprs: &[Expr], py: Python) -> Result<Vec<PyObject>, DataFusionError> {
    let ret: Result<Vec<PyObject>, DataFusionError> = exprs
        .iter()
        .map(|expr| match expr {
            Expr::Literal(v) => match v {
                ScalarValue::Boolean(Some(b)) => Ok(b.into_py(py)),
                ScalarValue::Int8(Some(i)) => Ok(i
                    .into_pyobject(py)
                    .map_err(to_datafusion_err)?
                    .into_any()
                    .unbind()),
                ScalarValue::Int16(Some(i)) => Ok(i
                    .into_pyobject(py)
                    .map_err(to_datafusion_err)?
                    .into_any()
                    .unbind()),
                ScalarValue::Int32(Some(i)) => Ok(i
                    .into_pyobject(py)
                    .map_err(to_datafusion_err)?
                    .into_any()
                    .unbind()),
                ScalarValue::Int64(Some(i)) => Ok(i
                    .into_pyobject(py)
                    .map_err(to_datafusion_err)?
                    .into_any()
                    .unbind()),
                ScalarValue::UInt8(Some(i)) => Ok(i
                    .into_pyobject(py)
                    .map_err(to_datafusion_err)?
                    .into_any()
                    .unbind()),
                ScalarValue::UInt16(Some(i)) => Ok(i
                    .into_pyobject(py)
                    .map_err(to_datafusion_err)?
                    .into_any()
                    .unbind()),
                ScalarValue::UInt32(Some(i)) => Ok(i
                    .into_pyobject(py)
                    .map_err(to_datafusion_err)?
                    .into_any()
                    .unbind()),
                ScalarValue::UInt64(Some(i)) => Ok(i
                    .into_pyobject(py)
                    .map_err(to_datafusion_err)?
                    .into_any()
                    .unbind()),
                ScalarValue::Float32(Some(f)) => Ok(f
                    .into_pyobject(py)
                    .map_err(to_datafusion_err)?
                    .into_any()
                    .unbind()),
                ScalarValue::Float64(Some(f)) => Ok(f
                    .into_pyobject(py)
                    .map_err(to_datafusion_err)?
                    .into_any()
                    .unbind()),
                ScalarValue::Utf8(Some(s)) => Ok(s
                    .into_pyobject(py)
                    .map_err(to_datafusion_err)?
                    .into_any()
                    .unbind()),
                _ => Err(DataFusionError::Common(format!(
                    "PyArrow can't handle ScalarValue: {v:?}"
                ))),
            },
            _ => Err(DataFusionError::Common(format!(
                "Only a list of Literals are supported got {expr:?}"
            ))),
        })
        .collect();
    ret
}

impl PyArrowFilterExpression {
    pub fn inner(&self) -> &PyObject {
        &self.0
    }
}

impl TryFrom<&Expr> for PyArrowFilterExpression {
    type Error = DataFusionError;

    fn try_from(expr: &Expr) -> Result<Self, Self::Error> {
        Python::with_gil(|py| {
            let pc = Python::import(py, "pyarrow.compute")?;
            let op_module = Python::import(py, "operator")?;
            let pc_expr: Result<Bound<'_, PyAny>, DataFusionError> = match expr {
                Expr::Column(Column { name, .. }) => Ok(pc.getattr("field")?.call1((name,))?),
                Expr::Literal(v) => match v {
                    ScalarValue::Boolean(Some(b)) => Ok(pc.getattr("scalar")?.call1((*b,))?),
                    ScalarValue::Int8(Some(i)) => Ok(pc.getattr("scalar")?.call1((*i,))?),
                    ScalarValue::Int16(Some(i)) => Ok(pc.getattr("scalar")?.call1((*i,))?),
                    ScalarValue::Int32(Some(i)) => Ok(pc.getattr("scalar")?.call1((*i,))?),
                    ScalarValue::Int64(Some(i)) => Ok(pc.getattr("scalar")?.call1((*i,))?),
                    ScalarValue::UInt8(Some(i)) => Ok(pc.getattr("scalar")?.call1((*i,))?),
                    ScalarValue::UInt16(Some(i)) => Ok(pc.getattr("scalar")?.call1((*i,))?),
                    ScalarValue::UInt32(Some(i)) => Ok(pc.getattr("scalar")?.call1((*i,))?),
                    ScalarValue::UInt64(Some(i)) => Ok(pc.getattr("scalar")?.call1((*i,))?),
                    ScalarValue::Float32(Some(f)) => Ok(pc.getattr("scalar")?.call1((*f,))?),
                    ScalarValue::Float64(Some(f)) => Ok(pc.getattr("scalar")?.call1((*f,))?),
                    ScalarValue::Utf8(Some(s)) => Ok(pc.getattr("scalar")?.call1((s,))?),
                    _ => Err(DataFusionError::Common(format!(
                        "PyArrow can't handle ScalarValue: {v:?}"
                    ))),
                },
                Expr::BinaryExpr(BinaryExpr { left, op, right }) => {
                    let operator = operator_to_py(op, &op_module)?;
                    let left = PyArrowFilterExpression::try_from(left.as_ref())?.0;
                    let right = PyArrowFilterExpression::try_from(right.as_ref())?.0;
                    Ok(operator.call1((left, right))?)
                }
                Expr::Not(expr) => {
                    let operator = op_module.getattr("invert")?;
                    let py_expr = PyArrowFilterExpression::try_from(expr.as_ref())?.0;
                    Ok(operator.call1((py_expr,))?)
                }
                Expr::IsNotNull(expr) => Ok(PyArrowFilterExpression::try_from(expr.as_ref())?
                    .0
                    .bind(py)
                    .call_method0("is_valid")?),
                Expr::IsNull(expr) => Ok(PyArrowFilterExpression::try_from(expr.as_ref())?
                    .0
                    .bind(py)
                    .call_method0("is_null")?),
                Expr::Between(Between {
                    expr,
                    negated,
                    low,
                    high,
                }) => {
                    let expr = PyArrowFilterExpression::try_from(expr.as_ref())?.0;
                    let low = PyArrowFilterExpression::try_from(low.as_ref())?.0;
                    let high = PyArrowFilterExpression::try_from(high.as_ref())?.0;
                    let and = op_module.getattr("and_")?;
                    let le = op_module.getattr("le")?;
                    let invert = op_module.getattr("invert")?;

                    // scalar <= field() returns a boolean expression so we need to use and to combine these
                    let ret = and.call1((
                        le.call1((low, expr.clone_ref(py)))?,
                        le.call1((expr, high))?,
                    ))?;

                    Ok(if *negated { invert.call1((ret,))? } else { ret })
                }
                Expr::InList(InList {
                    expr,
                    list,
                    negated,
                }) => {
                    let scalars = extract_scalar_list(list, py)?;
                    let invert = op_module.getattr("invert")?;
                    PyArrowFilterExpression::try_from(expr.as_ref())?
                        .0
                        .bind(py)
                        .call_method1("isin", (scalars,))
                        .map(|ret| {
                            if *negated {
                                invert.call1((ret,)).unwrap()
                            } else {
                                ret
                            }
                        })
                        .map_err(DataFusionError::from)
                }
                _ => Err(DataFusionError::Common(format!(
                    "Unsupported Datafusion expression {expr:?}"
                ))),
            };
            Ok(PyArrowFilterExpression(pc_expr?.into()))
        })
    }
}
