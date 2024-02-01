use std::any::Any;
use std::sync::Arc;

use arrow::array::{Array, Float32Array};
use datafusion::error::{DataFusionError, Result};
use datafusion::{
    arrow::{array::Float64Array, datatypes::DataType},
    logical_expr::Volatility,
};
use datafusion_common::ScalarValue;
use datafusion_expr::{ColumnarValue, FuncMonotonicity, ScalarUDFImpl, Signature};
use gbdt::decision_tree::Data;
use gbdt::gradient_boost::GBDT;

#[derive(Debug, Clone)]
pub(crate) struct PredictUdf {
    signature: Signature,
    aliases: Vec<String>,
}

impl PredictUdf {
    /// Create a new instance of the `PredictUdf` struct
    pub(crate) fn new() -> Self {
        Self {
            signature: Signature::variadic_any(Volatility::Immutable),
            // we will also add an alias of "my_pow"
            aliases: vec!["predict_xgb".to_string()],
        }
    }
}

impl ScalarUDFImpl for PredictUdf {
    /// We implement as_any so that we can downcast the ScalarUDFImpl trait object
    fn as_any(&self) -> &dyn Any {
        self
    }

    /// Return the name of this function
    fn name(&self) -> &str {
        "predict_xgb"
    }

    /// Return the "signature" of this function -- namely what types of arguments it will take
    fn signature(&self) -> &Signature {
        &self.signature
    }

    fn return_type(&self, _arg_types: &[DataType]) -> Result<DataType> {
        Ok(DataType::Float32)
    }

    /// This is the function that actually calculates the results.
    fn invoke(&self, args: &[ColumnarValue]) -> Result<ColumnarValue> {
        let model_path = &args[0];
        let objective = &args[1];
        assert_eq!(model_path.data_type(), DataType::Utf8);
        assert_eq!(objective.data_type(), DataType::Utf8);

        let model = match (model_path, objective) {
            (
                ColumnarValue::Scalar(ScalarValue::Utf8(path)),
                ColumnarValue::Scalar(ScalarValue::Utf8(obj)),
            ) => Some(
                GBDT::from_xgboost_dump(path.as_ref().unwrap(), obj.as_ref().unwrap())
                    .expect("failed to load model"),
            ),
            _ => None,
        };

        let mut result = Vec::new();
        let mut rows: Vec<Vec<f32>> = Vec::new();
        for arg in args.iter().skip(2) {
            let array = &arg.clone().into_array(0).unwrap();
            let values = array
                .as_any()
                .downcast_ref::<Float64Array>()
                .ok_or_else(|| DataFusionError::Internal("Expected Float64Array".to_string()))
                .unwrap();
            if rows.is_empty() {
                for i in 0..values.len() {
                    let single: Vec<f32> = vec![values.value(i) as f32];
                    rows.push(single)
                }
            } else {
                for i in 0..values.len() {
                    let x = values.value(i) as f32;
                    rows.get_mut(i).unwrap().push(x)
                }
            }
        }

        for row in rows {
            result.push(Data::new_test_data(row, None));
        }

        let predictions = model.unwrap().predict(&result);
        let res = Float32Array::from(predictions);
        Ok(ColumnarValue::Array(Arc::new(res)))
    }

    fn aliases(&self) -> &[String] {
        &self.aliases
    }

    fn monotonicity(&self) -> Result<Option<FuncMonotonicity>> {
        Ok(Some(vec![Some(true)]))
    }
}
