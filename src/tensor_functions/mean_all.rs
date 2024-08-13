use std::{any::Any, sync::Arc};

use arrow::array::{Array, ArrayRef, Float64Array};
use arrow::datatypes::DataType;
use arrow_convert::deserialize::TryIntoCollection;
use arrow_schema::DataType::{FixedSizeList, LargeList, List};
use candle_core::{Device, Tensor};
use datafusion_common::cast::as_list_array;
use datafusion_common::{exec_err, Result};
use datafusion_expr::{ColumnarValue, ScalarUDFImpl, Signature, Volatility};
use itertools::Itertools;

use crate::utils::make_scalar_function;

#[derive(Debug, Clone)]
pub(crate) struct TensorMeanAllUDF {
    signature: Signature,
    aliases: Vec<String>,
}

impl TensorMeanAllUDF {
    pub(crate) fn new() -> Self {
        Self {
            signature: Signature::variadic_any(Volatility::Immutable),
            aliases: vec!["tensor_avg_all".to_string()],
        }
    }
}

impl ScalarUDFImpl for TensorMeanAllUDF {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn name(&self) -> &str {
        "tensor_mean_all"
    }

    fn signature(&self) -> &Signature {
        &self.signature
    }

    fn return_type(&self, arg_types: &[DataType]) -> Result<DataType> {
        match &arg_types[0] {
            List(field) | FixedSizeList(field, _) | LargeList(field) => match field.data_type() {
                List(nested) | FixedSizeList(nested, _) | LargeList(nested) => {
                    Ok(nested.data_type().clone())
                }
                _ => Ok(field.data_type().clone()),
            },
            _ => exec_err!("Not reachable, data_type should be List, LargeList or FixedSizeList"),
        }
    }

    fn invoke(&self, _args: &[ColumnarValue]) -> Result<ColumnarValue> {
        make_scalar_function(tensor_mean_all)(_args)
    }

    fn aliases(&self) -> &[String] {
        &self.aliases
    }
}

enum Dimensions {
    D1,
    D2,
}

pub fn tensor_mean_all(args: &[ArrayRef]) -> Result<ArrayRef> {
    if args.len() != 1 {
        return exec_err!("tensor_mean_all needs one argument");
    }

    let extract = match args[0].data_type() {
        List(field) | LargeList(field) | FixedSizeList(field, _) => match field.data_type() {
            List(_) | FixedSizeList(_, _) | LargeList(_) => Ok(Dimensions::D2),
            _ => Ok(Dimensions::D1),
        },
        _ => exec_err!("Not reachable, data_type should be List, LargeList or FixedSizeList"),
    }?;

    let list_array = as_list_array(&args[0])?;
    let row_count = list_array.len();

    let device = Device::Cpu;
    let mut array_mean = vec![];

    for i in 0..row_count {
        if list_array.is_null(i) {
            array_mean.push(0.0);
        } else {
            let (rows, cols, data) = match &extract {
                Dimensions::D1 => extract1d(list_array.value(i))?,
                Dimensions::D2 => extract2d(list_array.value(i))?,
            };
            let a = Tensor::from_vec(data, (rows, cols), &device).unwrap();
            let a = a.mean_all().unwrap().to_scalar::<f64>();
            array_mean.push(a.unwrap());
        }
    }

    Ok(Arc::new(Float64Array::from(array_mean)))
}

fn extract1d(arr_ref: ArrayRef) -> Result<(usize, usize, Vec<f64>)> {
    let arr_vec: Vec<f64> = arr_ref.try_into_collection().unwrap();
    let elements = arr_vec.len();
    Ok((1, elements, arr_vec))
}

fn extract2d(arr_ref: ArrayRef) -> Result<(usize, usize, Vec<f64>)> {
    let arr_vec: Vec<Vec<Option<f64>>> = arr_ref.try_into_collection().unwrap();
    let unique: Vec<usize> = arr_vec.iter().map(|v| v.len()).unique().collect();
    if unique.len() > 1 {
        return exec_err!("jagged array");
    }
    let rows = arr_vec.len();
    let cols = unique[0];

    let arr_vec: Vec<f64> = arr_vec
        .into_iter()
        .flatten()
        .map(|e| e.unwrap_or_default())
        .collect();
    Ok((rows, cols, arr_vec))
}
