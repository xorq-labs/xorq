use datafusion_common::{Column, ScalarValue};
use datafusion_expr::expr::InList;
use datafusion_expr::{Between, BinaryExpr, Expr, Operator};
use pyo3::prelude::PyModule;
use pyo3::{IntoPy, PyAny, PyObject, Python};

use crate::errors::DataFusionError;

#[derive(Debug, Clone)]
#[repr(transparent)]
pub(crate) struct IbisFilterExpression(PyObject);

fn operator_to_py<'py>(
    operator: &Operator,
    op: &'py PyModule,
) -> Result<&'py PyAny, DataFusionError> {
    let py_op: &PyAny = match operator {
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
                ScalarValue::Int8(Some(i)) => Ok(i.into_py(py)),
                ScalarValue::Int16(Some(i)) => Ok(i.into_py(py)),
                ScalarValue::Int32(Some(i)) => Ok(i.into_py(py)),
                ScalarValue::Int64(Some(i)) => Ok(i.into_py(py)),
                ScalarValue::UInt8(Some(i)) => Ok(i.into_py(py)),
                ScalarValue::UInt16(Some(i)) => Ok(i.into_py(py)),
                ScalarValue::UInt32(Some(i)) => Ok(i.into_py(py)),
                ScalarValue::UInt64(Some(i)) => Ok(i.into_py(py)),
                ScalarValue::Float32(Some(f)) => Ok(f.into_py(py)),
                ScalarValue::Float64(Some(f)) => Ok(f.into_py(py)),
                ScalarValue::Utf8(Some(s)) => Ok(s.into_py(py)),
                _ => Err(DataFusionError::Common(format!(
                    "Ibis can't handle ScalarValue: {v:?}"
                ))),
            },
            _ => Err(DataFusionError::Common(format!(
                "Only a list of Literals are supported got {expr:?}"
            ))),
        })
        .collect();
    ret
}

impl IbisFilterExpression {
    pub fn inner(&self) -> &PyObject {
        &self.0
    }
}

impl TryFrom<&Expr> for IbisFilterExpression {
    type Error = DataFusionError;

    fn try_from(expr: &Expr) -> Result<Self, Self::Error> {
        Python::with_gil(|py| {
            let ibis = Python::import(py, "ibis")?;
            let op_module = Python::import(py, "operator")?;
            let deferred = ibis.getattr("_")?;

            let ibis_expr: Result<&PyAny, DataFusionError> = match expr {
                Expr::Column(Column { name, .. }) => Ok(deferred.getattr(name.as_str())?),
                Expr::Literal(v) => match v {
                    ScalarValue::Boolean(Some(b)) => Ok(ibis.getattr("literal")?.call1((*b,))?),
                    ScalarValue::Int8(Some(i)) => Ok(ibis.getattr("literal")?.call1((*i,))?),
                    ScalarValue::Int16(Some(i)) => Ok(ibis.getattr("literal")?.call1((*i,))?),
                    ScalarValue::Int32(Some(i)) => Ok(ibis.getattr("literal")?.call1((*i,))?),
                    ScalarValue::Int64(Some(i)) => Ok(ibis.getattr("literal")?.call1((*i,))?),
                    ScalarValue::UInt8(Some(i)) => Ok(ibis.getattr("literal")?.call1((*i,))?),
                    ScalarValue::UInt16(Some(i)) => Ok(ibis.getattr("literal")?.call1((*i,))?),
                    ScalarValue::UInt32(Some(i)) => Ok(ibis.getattr("literal")?.call1((*i,))?),
                    ScalarValue::UInt64(Some(i)) => Ok(ibis.getattr("literal")?.call1((*i,))?),
                    ScalarValue::Float32(Some(f)) => Ok(ibis.getattr("literal")?.call1((*f,))?),
                    ScalarValue::Float64(Some(f)) => Ok(ibis.getattr("literal")?.call1((*f,))?),
                    ScalarValue::Utf8(Some(s)) => Ok(ibis.getattr("literal")?.call1((s,))?),
                    _ => Err(DataFusionError::Common(format!(
                        "Ibis can't handle ScalarValue: {v:?}"
                    ))),
                },
                Expr::BinaryExpr(BinaryExpr { left, op, right }) => {
                    let operator = operator_to_py(op, op_module)?;
                    let left = IbisFilterExpression::try_from(left.as_ref())?.0;
                    let right = IbisFilterExpression::try_from(right.as_ref())?.0;
                    Ok(operator.call1((left, right))?)
                }
                Expr::Not(expr) => {
                    let operator = op_module.getattr("invert")?;
                    let py_expr = IbisFilterExpression::try_from(expr.as_ref())?.0;
                    Ok(operator.call1((py_expr,))?)
                }
                Expr::IsNotNull(expr) => {
                    let py_expr = IbisFilterExpression::try_from(expr.as_ref())?
                        .0
                        .into_ref(py);
                    Ok(py_expr.call_method0("notnull")?)
                }
                Expr::IsNull(expr) => {
                    let expr = IbisFilterExpression::try_from(expr.as_ref())?
                        .0
                        .into_ref(py);
                    Ok(expr.call_method0("isnull")?)
                }
                Expr::InList(InList {
                    expr,
                    list,
                    negated,
                }) => {
                    let expr = IbisFilterExpression::try_from(expr.as_ref())?
                        .0
                        .into_ref(py);
                    let scalars = extract_scalar_list(list, py)?;
                    let ret = if *negated {
                        expr.call_method1("isin", (scalars,))?
                    } else {
                        expr.call_method1("notin", (scalars,))?
                    };
                    Ok(ret)
                }
                Expr::Between(Between {
                    expr,
                    negated,
                    low,
                    high,
                }) => {
                    let expr = IbisFilterExpression::try_from(expr.as_ref())?
                        .0
                        .into_ref(py);
                    let low = IbisFilterExpression::try_from(low.as_ref())?.0;
                    let high = IbisFilterExpression::try_from(high.as_ref())?.0;
                    let invert = op_module.getattr("invert")?;
                    let ret = expr.call_method1("between", (low, high))?;
                    Ok(if *negated { invert.call1((ret,))? } else { ret })
                }
                _ => Err(DataFusionError::Common(format!(
                    "Unsupported Datafusion expression {expr:?}"
                ))),
            };

            Ok(IbisFilterExpression(ibis_expr?.into()))
        })
    }
}
