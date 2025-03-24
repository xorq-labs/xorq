use pyo3::prelude::*;

use crate::errors::DataFusionError;
use crate::pyarrow_filter_expression::extract_scalar_list;
use datafusion_common::{Column, ScalarValue};
use datafusion_expr::expr::InList;
use datafusion_expr::{Between, BinaryExpr, Expr, Operator};

#[derive(Debug)]
#[repr(transparent)]
pub(crate) struct IbisFilterExpression(PyObject);

fn operator_to_py<'py>(
    operator: &Operator,
    op: &Bound<'py, PyModule>,
) -> Result<Bound<'py, PyAny>, DataFusionError> {
    let py_op: Bound<'py, PyAny> = match operator {
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

            let ibis_expr: Result<Bound<'_, PyAny>, DataFusionError> = match expr {
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
                    let operator = operator_to_py(op, &op_module)?;
                    let left = IbisFilterExpression::try_from(left.as_ref())?.0;
                    let right = IbisFilterExpression::try_from(right.as_ref())?.0;
                    Ok(operator.call1((left, right))?)
                }
                Expr::Not(expr) => {
                    let operator = op_module.getattr("invert")?;
                    let py_expr = IbisFilterExpression::try_from(expr.as_ref())?.0;
                    Ok(operator.call1((py_expr,))?)
                }
                Expr::IsNotNull(expr) => Ok(IbisFilterExpression::try_from(expr.as_ref())?
                    .0
                    .bind(py)
                    .call_method0("notnull")?),
                Expr::IsNull(expr) => Ok(IbisFilterExpression::try_from(expr.as_ref())?
                    .0
                    .bind(py)
                    .call_method0("isnull")?),
                Expr::InList(InList {
                    expr,
                    list,
                    negated,
                }) => {
                    let scalars = extract_scalar_list(list, py)?;
                    Ok(if *negated {
                        IbisFilterExpression::try_from(expr.as_ref())?
                            .0
                            .bind(py)
                            .call_method1("isin", (scalars,))?
                    } else {
                        IbisFilterExpression::try_from(expr.as_ref())?
                            .0
                            .bind(py)
                            .call_method1("notin", (scalars,))?
                    })
                }
                Expr::Between(Between {
                    expr,
                    negated,
                    low,
                    high,
                }) => {
                    let low = IbisFilterExpression::try_from(low.as_ref())?.0;
                    let high = IbisFilterExpression::try_from(high.as_ref())?.0;
                    let ret = IbisFilterExpression::try_from(expr.as_ref())?
                        .0
                        .bind(py)
                        .call_method1("between", (low, high))?;
                    let invert = op_module.getattr("invert")?;
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
