//! Code for loading and running (trained) PyTorch models

use crate::rpc;
use anyhow::{anyhow, Result};
use base64::{
    alphabet,
    engine::{self, general_purpose},
    Engine as _,
};

use image::{Cursor, DynamicImage, ImageBuffer, ImageOutputFormat, PngEncoder, Rgb};
use serde::Serialize;
use tch::vision::imagenet;
use tch::{no_grad, vision, Device, IValue, Kind, Tensor};

/// An in-memory representation of an image. Can be the input or output of a model
#[derive(Debug, Serialize, Clone)]
pub struct Image {
    pub(crate) image: Vec<u8>,
    pub(crate) height: Option<u32>,
    pub(crate) width: Option<u32>,
}

/// A class prediction outputted by a classifier model
#[derive(Debug, Serialize)]
pub struct Class {
    probability: Option<f64>,
    class: Option<i32>,
    label: Option<String>,
}

/// The output of a model's inference
#[derive(Debug, Serialize)]
pub enum Inference {
    Text(String),
    Classification(Vec<Class>),
    Image(Image),
    B64Image(String),
}

#[derive(Debug, Clone)]
/// The type of inference to compute
pub enum InferenceType {
    /// `InputData::Image` to `Inference::Classification`
    ImageClassification { top_n: u16 },

    /// `InputData::Image` to `Inference::Image`. Used for object detection
    /// or image segmentation
    ImageToImage,

    /// `InputData::Text` to `Inference::Text`, for NLP tasks
    SentimentAnalysis,
}

/// Input data that inference can be computed on
#[derive(Debug, Clone)]
pub enum InputData {
    Text(String),
    Image(Image),
}

/// The input to this module's ML engine -- a request for inference
pub struct InferenceTask {
    data: InputData,
    inference_type: InferenceType,
}

/// Load and run a TorchScript file
#[derive(Debug)]
pub struct TorchModel {
    /// The loaded torch model
    model: tch::jit::CModule,
}

impl TorchModel {
    pub fn new(filename: String) -> Result<Self> {
        Ok(TorchModel {
            model: tch::CModule::load(filename)?,
        })
    }

    /// Run image classification
    fn image_classification(&self, image: Image, top_n: u16) -> Result<Inference> {
        let image = imagenet::load_image_from_memory(&image.image)?;
        let output = self
            .model
            .forward_ts(&[image.unsqueeze(0)])?
            .softmax(-1, Some(tch::kind::Kind::Float));
        let classes = imagenet::top(&output, top_n as i64)
            .iter()
            .map(|(p, l)| Class {
                probability: Some(*p),
                class: None,
                label: Some(l.into()),
            })
            .collect();
        Ok(Inference::Classification(classes))
    }

    /// Run image-to-image inference
    fn image_to_image(&self, image: Image) -> Result<Inference> {
        let img = imagenet::load_image_from_memory(&image.image)?;

        // Convert the image to a Tensor and normalize it
        let img: Tensor = img.into();
        let img = img.to_kind(Kind::Float) / 255.; // normalize to [0, 1]
        let img = img.permute(&[2, 0, 1]); // from [height, width, channels] to [channels, height, width]
        let img = img.unsqueeze(0); // add batch dimension

        // Run the model on the image
        let output = no_grad(|| self.model.forward_is(&[img.into()]))?;

        // The output is a Tensor with shape [1, num_classes, height, width]
        // You can convert it to a 2D image where each pixel's value is the class index
        let output = match output {
            IValue::Tensor(t) => t,
            _ => return Err(anyhow!("invalid type")),
        };
        //let output: Tensor = output.get(0);
        let output = output.squeeze();
        let output = output.argmax(0, false); // get the class index for each pixel
        let output = output.to_kind(Kind::Uint8); // convert to uint8

        // Convert the Tensor back to an ImageBuffer
        let (width, height) = (output.size()[1], output.size()[0]);
        //let output_data: Vec<u8> = output.into();
        let output_data = output.view([-1]).to_kind(tch::Kind::Uint8).into();

        let output_image: ImageBuffer<Rgb<u8>, _> =
            ImageBuffer::from_raw(width as u32, height as u32, output_data).unwrap();

        // Create a Vec<u8> to write to
        let mut buffer: Vec<u8> = Vec::new();

        // Write the image to the buffer in PNG format
        {
            let encoder = PngEncoder::new(&mut buffer);
            encoder
                .encode(
                    &output_image,
                    output_image.width(),
                    output_image.height(),
                    image::ColorType::Rgb8,
                )
                .unwrap();
        }

        // Encode the buffer as base-64
        let b64_image = general_purpose::STANDARD.encode(&buffer);

        Ok(Inference::B64Image(b64_image))
    }

    /// Run inference on the loaded model given an `InferenceTask`
    pub fn run(&self, task: InferenceTask) -> Result<Inference> {
        match task.inference_type {
            InferenceType::ImageClassification { top_n } => match task.data {
                InputData::Image(image) => Ok(self.image_classification(image, top_n)?),
                _ => Err(anyhow!(
                    "invalid input type for ImageClassification inference"
                )),
            },
            InferenceType::ImageToImage => match task.data {
                InputData::Image(image) => Ok(self.image_to_image(image)?),
                _ => Err(anyhow!("invalid input type for ImageToImage inference")),
            },
            _ => Err(anyhow!("that inference type is not currently supported")),
        }
    }
}

impl From<rpc::ImageInput> for InputData {
    fn from(img: rpc::ImageInput) -> Self {
        InputData::Image(Image {
            image: img.image,
            height: Some(img.height).filter(|x| *x != 0),
            width: Some(img.width).filter(|x| *x != 0),
        })
    }
}

impl From<InputData> for rpc::ImageInput {
    fn from(input: InputData) -> Self {
        match input {
            InputData::Image(img) => Self {
                image: img.image,
                height: img.height.unwrap_or(0),
                width: img.width.unwrap_or(0),
            },
            InputData::Text(_) => todo!(),
        }
    }
}

impl From<rpc::ClassOutput> for Inference {
    fn from(data: rpc::ClassOutput) -> Self {
        Inference::Classification(
            data.classes
                .into_iter()
                .map(|c| Class {
                    probability: Some(c.probability),
                    class: Some(c.class_int),
                    label: Some(c.label).filter(|x| x != ""),
                })
                .collect(),
        )
    }
}

impl From<Inference> for rpc::ClassOutput {
    fn from(data: Inference) -> Self {
        match data {
            Inference::Classification(c) => {
                let classes: Vec<rpc::Classification> = c
                    .into_iter()
                    .map(|c| rpc::Classification {
                        probability: c.probability.unwrap_or(-1.0),
                        label: c.label.unwrap_or(String::new()),
                        class_int: c.class.unwrap_or(-1),
                    })
                    .collect();
                rpc::ClassOutput {
                    num_classes: classes.len() as u32,
                    classes,
                }
            }
            _ => todo!(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::util::test;

    #[test]
    fn test_resnet18() {
        let loader = TorchModel::new("models/resnet18.pt".into()).unwrap();
        let task = InferenceTask {
            data: test::get_test_image(),
            inference_type: InferenceType::ImageClassification { top_n: 2 },
        };
        let outputs = loader.run(task).unwrap();
        println!("outputs: {outputs:#?}");
    }

    #[test]
    fn test_resnet50() {
        let loader = TorchModel::new("models/resnet50.pt".into()).unwrap();
        vec!["images/lamp.jpg", "images/cocoa.jpg"]
            .into_iter()
            .for_each(|img| {
                let task = InferenceTask {
                    data: test::load_image_from_disk(img.into()),
                    inference_type: InferenceType::ImageClassification { top_n: 2 },
                };
                let outputs = loader.run(task).unwrap();
                println!("outputs: {outputs:#?}");
            });
    }
}
