use std::any::Any;
use std::io::Cursor;
use std::sync::Arc;

use arrow::array::{Array, ArrayRef, LargeBinaryArray};
use arrow_convert::deserialize::TryIntoCollection;
use arrow_schema::DataType;
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
        Ok(DataType::LargeBinary)
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

    for (i, mut bytes) in (0..row_count).zip(segmented.clone()) {
        let image = images.value(i);
        let (format, mut image) = binary_to_img(image)?;
        let tensor = get_tensor_from_image(Some(sam::IMAGE_SIZE), image.clone())
            .map_err(|e| Execution(e.to_string()))?;

        let (mask, _iou_predictions) = sam
            .forward(&tensor, &points, false)
            .map_err(|e| Execution(e.to_string()))?;

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
        let mask_img: image::ImageBuffer<image::Rgb<u8>, Vec<u8>> =
            match image::ImageBuffer::from_raw(w as u32, h as u32, mask_pixels) {
                Some(image) => image,
                None => return exec_err!("error saving merged image"),
            };
        let mask_img = image::DynamicImage::from(mask_img).resize_to_fill(
            image.width(),
            image.height(),
            image::imageops::FilterType::CatmullRom,
        );
        for x in 0..image.width() {
            for y in 0..image.height() {
                let mask_p = imageproc::drawing::Canvas::get_pixel(&mask_img, x, y);
                if mask_p.0[0] > 100 {
                    let mut img_p = imageproc::drawing::Canvas::get_pixel(&image, x, y);
                    img_p.0[2] = 255 - (255 - img_p.0[2]) / 2;
                    img_p.0[1] /= 2;
                    img_p.0[0] /= 2;
                    imageproc::drawing::Canvas::draw_pixel(&mut image, x, y, img_p)
                }
            }
        }
        for (x, y, b) in &points {
            let x = (x * image.width() as f64) as i32;
            let y = (y * image.height() as f64) as i32;
            let color = if *b {
                image::Rgba([255, 0, 0, 200])
            } else {
                image::Rgba([0, 255, 0, 200])
            };
            imageproc::drawing::draw_filled_circle_mut(&mut image, (x, y), 3, color);
        }

        let mut writer = Cursor::new(&mut bytes);

        let _ = image.write_to(&mut writer, format);

        segmented[i] = bytes;
    }

    let result = segmented.iter().map(|v| v.as_slice()).collect();

    Ok(Arc::new(LargeBinaryArray::from_vec(result)))
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
