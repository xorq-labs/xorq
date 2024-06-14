use std::any::Any;
use std::sync::Arc;

use arrow::array::Array;
use datafusion::error::{DataFusionError, Result};
use datafusion::{
    arrow::{array::Float64Array, datatypes::DataType},
    logical_expr::Volatility,
};
use datafusion_common::ScalarValue;
use datafusion_expr::{ColumnarValue, ScalarUDFImpl, Signature};
use gbdt::decision_tree::Data;
use parking_lot::RwLock;

use crate::model::SessionModelRegistry;

#[derive(Debug, Clone)]
pub(crate) struct PredictUdf {
    signature: Signature,
    aliases: Vec<String>,
    model_registry: Arc<RwLock<SessionModelRegistry>>,
}

impl PredictUdf {
    /// Create a new instance of the `PredictUdf` struct
    pub(crate) fn new_with_model_registry(
        model_registry: Arc<RwLock<SessionModelRegistry>>,
    ) -> Self {
        Self {
            signature: Signature::variadic_any(Volatility::Immutable),
            // we will also add an alias of "my_pow"
            aliases: vec!["predict_xgb".to_string()],
            model_registry,
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
        Ok(DataType::Float64)
    }

    /// This is the function that actually calculates the results.
    fn invoke(&self, args: &[ColumnarValue]) -> Result<ColumnarValue> {
        let model_path = &args[0];
        assert_eq!(model_path.data_type(), DataType::Utf8);
        let model_registry = self.model_registry.read();
        let model = match model_path {
            ColumnarValue::Scalar(ScalarValue::Utf8(path)) => {
                model_registry.models.get(path.as_ref().unwrap().as_str())
            }
            _ => None,
        };

        let mut result = Vec::new();
        let mut rows: Vec<Vec<f64>> = Vec::new();
        for arg in args.iter().skip(1) {
            let array = &arg.clone().into_array(0).unwrap();
            let values = array
                .as_any()
                .downcast_ref::<Float64Array>()
                .ok_or_else(|| DataFusionError::Internal("Expected Float64Array".to_string()))
                .unwrap();
            if rows.is_empty() {
                for i in 0..values.len() {
                    let single: Vec<f64> = vec![values.value(i)];
                    rows.push(single)
                }
            } else {
                for i in 0..values.len() {
                    let x = values.value(i);
                    rows.get_mut(i).unwrap().push(x)
                }
            }
        }

        for row in rows {
            result.push(Data::new_test_data(row, None));
        }

        let predictions = model.unwrap().predict(&result);
        let res = Float64Array::from(predictions);
        Ok(ColumnarValue::Array(Arc::new(res)))
    }

    fn aliases(&self) -> &[String] {
        &self.aliases
    }
}
