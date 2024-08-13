use std::any::Any;
use std::io::Cursor;
use std::sync::Arc;

use arrow::array::{Array, ArrayRef, LargeBinaryArray};
use arrow_schema::DataType;
use datafusion_common::cast::as_binary_array;
use datafusion_common::{exec_err, Result};
use datafusion_expr::{ColumnarValue, ScalarUDFImpl, Signature, Volatility};

use crate::utils::{binary_to_img, make_scalar_function};

#[derive(Debug, Clone)]
pub(crate) struct Rotate90UDF {
    signature: Signature,
    aliases: Vec<String>,
}

impl Rotate90UDF {
    pub(crate) fn new() -> Self {
        Self {
            signature: Signature::variadic_any(Volatility::Immutable),
            aliases: vec!["image_rotate90".to_string()],
        }
    }
}

impl ScalarUDFImpl for Rotate90UDF {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn name(&self) -> &str {
        "rotate90"
    }

    fn signature(&self) -> &Signature {
        &self.signature
    }

    fn return_type(&self, _arg_types: &[DataType]) -> Result<DataType> {
        Ok(DataType::LargeBinary)
    }

    fn invoke(&self, args: &[ColumnarValue]) -> Result<ColumnarValue> {
        make_scalar_function(rotate90_inner)(args)
    }

    fn aliases(&self) -> &[String] {
        &self.aliases
    }
}

fn rotate90_inner(args: &[ArrayRef]) -> Result<ArrayRef> {
    if args.len() != 1 {
        return exec_err!("rotate90 needs 1 arguments");
    }

    let images = as_binary_array(&args[0])?;
    let row_count = images.len();
    let mut rotated: Vec<Vec<u8>> = (0..row_count).map(|_| Vec::new()).collect();

    for (i, mut bytes) in (0..row_count).zip(rotated.clone()) {
        let image = images.value(i);
        let (format, image) = binary_to_img(image)?;

        let mut writer = Cursor::new(&mut bytes);

        let _ = image.rotate90().write_to(&mut writer, format);

        rotated[i] = bytes;
    }

    let result = rotated.iter().map(|v| v.as_slice()).collect();

    Ok(Arc::new(LargeBinaryArray::from_vec(result)))
}
