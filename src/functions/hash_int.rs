use arrow::array::{Array, AsArray, Int64Array};
use arrow::datatypes::DataType;
use datafusion::logical_expr::{ColumnarValue, Signature, Volatility};
use datafusion_common::exec_err;
use datafusion_expr::ScalarUDFImpl;
use std::any::Any;
use std::hash::{DefaultHasher, Hash, Hasher};
use std::sync::Arc;

#[derive(Debug)]
pub struct HashIntFunc {
    signature: Signature,
}

impl Default for HashIntFunc {
    fn default() -> Self {
        HashIntFunc::new()
    }
}

impl HashIntFunc {
    pub fn new() -> Self {
        Self {
            signature: Signature::string(1, Volatility::Immutable),
        }
    }
}

impl ScalarUDFImpl for HashIntFunc {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn name(&self) -> &str {
        "hash_int"
    }

    fn signature(&self) -> &Signature {
        &self.signature
    }

    fn return_type(&self, _arg_types: &[DataType]) -> datafusion_common::Result<DataType> {
        Ok(DataType::Int64)
    }

    fn invoke(&self, args: &[ColumnarValue]) -> datafusion_common::Result<ColumnarValue> {
        if args.len() != 1 {
            return exec_err!(
                "{:?} args were supplied but encode takes exactly two arguments",
                args.len()
            );
        }

        let arrays = ColumnarValue::values_to_arrays(args)?;
        let mut result: Vec<i64> = Vec::new();

        match arrays[0].data_type() {
            DataType::Utf8 => {
                for maybe_val in arrays[0].as_string::<i32>().iter() {
                    let mut hasher = DefaultHasher::new();
                    maybe_val.hash(&mut hasher);
                    result.push(hasher.finish() as i64)
                }
            }
            DataType::LargeUtf8 => {
                for maybe_val in arrays[0].as_string::<i64>().iter() {
                    let mut hasher = DefaultHasher::new();
                    maybe_val.hash(&mut hasher);
                    result.push(hasher.finish() as i64)
                }
            }
            DataType::Utf8View => {
                for maybe_val in arrays[0].as_string_view().iter() {
                    let mut hasher = DefaultHasher::new();
                    maybe_val.hash(&mut hasher);
                    result.push(hasher.finish() as i64)
                }
            }
            _ => return exec_err!("Unexpected Datatype"),
        }

        let res = Int64Array::from(result);
        Ok(ColumnarValue::Array(Arc::new(res)))
    }
}
