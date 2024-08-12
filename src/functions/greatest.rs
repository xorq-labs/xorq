use std::any::Any;

use arrow::compute::kernels::zip::zip;
use arrow::datatypes::DataType;
use arrow_ord::cmp::gt;

use crate::functions::min_max::choose_min_max;
use crate::functions::min_max::interval_min_max;
use crate::functions::min_max::min_max;
use crate::functions::min_max::typed_min_max;
use crate::functions::min_max::typed_min_max_float;
use crate::functions::min_max::typed_min_max_string;
use datafusion_common::{exec_err, internal_err, Result, ScalarValue};
use datafusion_expr::type_coercion::binary::type_union_resolution;
use datafusion_expr::ColumnarValue;
use datafusion_expr::{ScalarUDFImpl, Signature, Volatility};

pub fn max(lhs: &ScalarValue, rhs: &ScalarValue) -> Result<ScalarValue> {
    min_max!(lhs, rhs, max)
}

#[derive(Debug)]
pub struct GreatestFunc {
    signature: Signature,
}

impl Default for GreatestFunc {
    fn default() -> Self {
        GreatestFunc::new()
    }
}

impl GreatestFunc {
    pub fn new() -> Self {
        Self {
            signature: Signature::user_defined(Volatility::Immutable),
        }
    }
}

impl ScalarUDFImpl for GreatestFunc {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn name(&self) -> &str {
        "greatest"
    }

    fn signature(&self) -> &Signature {
        &self.signature
    }

    fn return_type(&self, arg_types: &[DataType]) -> Result<DataType> {
        Ok(arg_types[0].clone())
    }

    fn invoke(&self, args: &[ColumnarValue]) -> Result<ColumnarValue> {
        // do not accept 0 arguments.
        if args.is_empty() {
            return exec_err!(
                "greatest was called with {} arguments. It requires at least 1.",
                args.len()
            );
        } else if args.len() == 1 {
            return Ok(args[0].clone());
        }

        let mut return_array = args.iter().filter_map(|x| match x {
            ColumnarValue::Array(array) => Some(array.len()),
            _ => None,
        });

        if let Some(length) = return_array.next() {
            // there is at least one array argument
            let first_arg = match &args[0] {
                ColumnarValue::Array(array) => array.clone(),
                ColumnarValue::Scalar(scalar) => scalar.to_array_of_size(length).unwrap(),
            };
            args[1..]
                .iter()
                .map(|arg| match arg {
                    ColumnarValue::Array(array) => array.clone(),
                    ColumnarValue::Scalar(scalar) => scalar.to_array_of_size(length).unwrap(),
                })
                .try_fold(first_arg, |a, b| {
                    // mask will be true if cmp holds for a to be otherwise false
                    let mask = gt(&a, &b)?;
                    // then the zip can pluck values accordingly from a and b
                    let value = zip(&mask, &a, &b)?;
                    Ok(value)
                })
                .map(ColumnarValue::Array)
        } else {
            // all arguments are scalars
            let args: Vec<_> = args
                .iter()
                .map(|arg| match arg {
                    ColumnarValue::Array(_) => {
                        panic!("Internal error: all arguments should be scalars")
                    }
                    ColumnarValue::Scalar(scalar) => scalar.clone(),
                })
                .collect();

            let first_arg = args[0].clone();
            args[1..]
                .iter()
                .try_fold(first_arg, |a, b| max(&a, b))
                .map(ColumnarValue::Scalar)
        }
    }

    fn short_circuits(&self) -> bool {
        false
    }

    fn coerce_types(&self, arg_types: &[DataType]) -> Result<Vec<DataType>> {
        if arg_types.is_empty() {
            return exec_err!("greatest must have at least one argument");
        }
        let new_type =
            type_union_resolution(arg_types).unwrap_or(arg_types.first().unwrap().clone());
        Ok(vec![new_type; arg_types.len()])
    }
}
