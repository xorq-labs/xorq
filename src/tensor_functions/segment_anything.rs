use std::any::Any;
use std::sync::Arc;

use arrow::array::{Array, ArrayRef, Float32Array, LargeListArray, StructArray};
use arrow::datatypes::UInt8Type;
use arrow_convert::deserialize::TryIntoCollection;
use arrow_schema::{DataType, Field, Fields};
use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::segment_anything::sam;
use datafusion_common::cast::{as_binary_array, as_list_array, as_string_array};
use datafusion_common::DataFusionError::Execution;
use datafusion_common::{exec_err, Result};
use datafusion_expr::{ColumnarValue, ScalarUDFImpl, Signature, Volatility};
use image::DynamicImage;

use crate::utils::{binary_to_img, make_scalar_function};

#[derive(Debug, Clone)]
pub(crate) struct SegmentAnythingUDF {
    signature: Signature,
    aliases: Vec<String>,
}

impl SegmentAnythingUDF {
    pub(crate) fn new() -> Self {
        Self {
            signature: Signature::variadic_any(Volatility::Immutable),
            aliases: vec!["sam".to_string()],
        }
    }
}

impl ScalarUDFImpl for SegmentAnythingUDF {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn name(&self) -> &str {
        "segment_anything"
    }

    fn signature(&self) -> &Signature {
        &self.signature
    }

    fn return_type(&self, _arg_types: &[DataType]) -> Result<DataType> {
        let struct_fields = Fields::from(vec![
            Field::new(
                "mask",
                DataType::LargeList(new_arc_field("item", DataType::UInt8, true)),
                false,
            ),
            Field::new("iou_score", DataType::Float32, false),
        ]);
        Ok(DataType::Struct(struct_fields.clone()))
    }

    fn invoke(&self, args: &[ColumnarValue]) -> Result<ColumnarValue> {
        make_scalar_function(segment_anything_inner)(args)
    }

    fn aliases(&self) -> &[String] {
        &self.aliases
    }
}

fn segment_anything_inner(args: &[ArrayRef]) -> Result<ArrayRef> {
    if args.len() != 3 {
        return exec_err!("segment_anything needs 3 arguments");
    }

    let str_array = as_string_array(&args[0])?;

    let model = std::path::PathBuf::from(str_array.value(0));

    let device = Device::Cpu;

    let vb = unsafe {
        VarBuilder::from_mmaped_safetensors(&[model], DType::F32, &device)
            .map_err(|e| Execution(e.to_string()))?
    };
    let sam = sam::Sam::new_tiny(vb).map_err(|e| Execution(e.to_string()))?;

    let list_array = as_list_array(&args[2]);

    let seed: Vec<Option<f64>> = list_array.unwrap().value(0).try_into_collection()?;

    let points = [seed]
        .iter()
        .map(|v| {
            if v.len() != 2 {
                return exec_err!("expected format for points is 0.4,0.2");
            }

            let x = v[0].unwrap();
            let y = v[1].unwrap();

            Ok((x, y, true))
        })
        .collect::<Result<Vec<_>>>()
        .map_err(|e| Execution(e.to_string()))?;

    let images = as_binary_array(&args[1])?;
    let row_count = images.len();
    let mut segmented: Vec<Vec<u8>> = (0..row_count).map(|_| Vec::new()).collect();
    let mut iou_scores: Vec<f32> = Vec::new();

    for (i, image) in images.iter().enumerate().take(row_count) {
        let (_, image) = binary_to_img(image.unwrap_or_default())?;
        let tensor = get_tensor_from_image(Some(sam::IMAGE_SIZE), image.clone())
            .map_err(|e| Execution(e.to_string()))?;

        let (mask, _iou_predictions) = sam
            .forward(&tensor, &points, false)
            .map_err(|e| Execution(e.to_string()))?;

        let iou_score = _iou_predictions
            .flatten_all()
            .map_err(|e| Execution(e.to_string()))?
            .to_vec1::<f32>()
            .map_err(|e| Execution(e.to_string()))?[0];

        let mask = (mask.ge(0.).map_err(|e| Execution(e.to_string()))? * 255.)
            .map_err(|e| Execution(e.to_string()))?;
        let (_one, h, w) = mask.dims3().map_err(|e| Execution(e.to_string()))?;
        let mask = mask
            .expand((3, h, w))
            .map_err(|e| Execution(e.to_string()))?;

        let mask_pixels = mask
            .permute((1, 2, 0))
            .map_err(|e| Execution(e.to_string()))?
            .flatten_all()
            .map_err(|e| Execution(e.to_string()))?
            .to_vec1::<u8>()
            .map_err(|e| Execution(e.to_string()))?;

        segmented[i] = mask_pixels;
        iou_scores.push(iou_score);
    }

    let segmented: Vec<Option<Vec<Option<u8>>>> = segmented
        .iter()
        .map(|v| Some(v.iter().map(|x1| Some(*x1)).collect::<Vec<Option<u8>>>()))
        .collect();
    let result = Arc::new(LargeListArray::from_iter_primitive::<UInt8Type, _, _>(
        segmented,
    ));
    let iou_scores = Arc::new(Float32Array::from(iou_scores));

    let expected = StructArray::from(vec![
        (
            Arc::new(Field::new(
                "mask",
                DataType::LargeList(new_arc_field("item", DataType::UInt8, true)),
                false,
            )),
            Arc::clone(&result) as ArrayRef,
        ),
        (
            Arc::new(Field::new("iou_score", DataType::Float32, false)),
            Arc::clone(&iou_scores) as ArrayRef,
        ),
    ]);

    Ok(Arc::new(expected))
}

fn get_tensor_from_image(
    resize_longest: Option<usize>,
    img: DynamicImage,
) -> candle_core::Result<Tensor> {
    let img = match resize_longest {
        None => img,
        Some(resize_longest) => {
            let (height, width) = (img.height(), img.width());
            let resize_longest = resize_longest as u32;
            let (height, width) = if height < width {
                let h = (resize_longest * height) / width;
                (h, resize_longest)
            } else {
                let w = (resize_longest * width) / height;
                (resize_longest, w)
            };
            img.resize_exact(width, height, image::imageops::FilterType::CatmullRom)
        }
    };
    let (height, width) = (img.height() as usize, img.width() as usize);
    let img = img.to_rgb8();
    let data = img.into_raw();
    let data = Tensor::from_vec(data, (height, width, 3), &Device::Cpu)?.permute((2, 0, 1))?;
    Ok(data)
}

fn new_arc_field(name: &str, dt: DataType, nullable: bool) -> Arc<Field> {
    Arc::new(Field::new(name, dt, nullable))
}
