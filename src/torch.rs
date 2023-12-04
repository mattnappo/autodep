//! Code for loading and running (trained) PyTorch models

use crate::rpc;
use anyhow::{anyhow, Result};
use base64::{
    alphabet,
    engine::{self, general_purpose},
    Engine as _,
};

use base64;
use image::{codecs::png::PngEncoder, DynamicImage, ImageBuffer, ImageOutputFormat, Rgb};
use serde::{Deserialize, Serialize};
use std::io::Cursor;
use tch::vision::imagenet;
use tch::{no_grad, vision, Device, IValue, Kind, Tensor};

/// An in-memory representation of an image (not base 64). Can be the input or output of a model
#[derive(Debug, Serialize, Clone, Deserialize)]
pub struct Image {
    pub(crate) image: Vec<u8>,
    pub(crate) height: Option<u32>,
    pub(crate) width: Option<u32>,
}

/// A base 64 image
#[derive(Serialize, Deserialize, Debug)]
pub struct B64Image {
    pub image: String,
    pub height: Option<u32>,
    pub width: Option<u32>,
}

impl From<B64Image> for Image {
    fn from(b64_img: B64Image) -> Image {
        let image = base64::decode(&b64_img.image).unwrap_or_else(|_| Vec::new());
        Image {
            image,
            height: b64_img.height,
            width: b64_img.width,
        }
    }
}

impl From<rpc::B64Image> for B64Image {
    fn from(b64_img: rpc::B64Image) -> Self {
        B64Image {
            image: b64_img.image,
            height: b64_img.height,
            width: b64_img.width,
        }
    }
}

impl From<Image> for B64Image {
    fn from(img: Image) -> B64Image {
        let image = base64::encode(&img.image);
        B64Image {
            image,
            height: img.height,
            width: img.width,
        }
    }
}

#[derive(Deserialize)]
pub enum InferenceRequest {
    Image(B64Image),
    Text(String),
}

/// A class prediction outputted by a classifier model
#[derive(Debug, Serialize)]
pub struct Class {
    probability: Option<f64>,
    label: Option<String>,
}

/// The output of a model's inference
#[derive(Debug, Serialize)]
pub enum Inference {
    Text(String),
    Classification(Vec<Class>),
    //Image(Image),
    B64Image(B64Image),
}

#[derive(Deserialize, Debug, Clone)]
/// The type of inference to compute
pub enum InferenceType {
    /// `InputData::Image` to `Inference::Classification`
    ImageClassification { top_n: u16 },

    /// `InputData::Image` to `Inference::Image`. Used for object detection
    /// or image segmentation
    ImageToImage,

    /// `InputData::Text` to `Inference::Text`, for NLP tasks
    TextToText,
}

/// Input data that inference can be computed on
#[derive(Deserialize, Debug)]
pub enum InputData {
    Text(String),
    //Image(Image),
    B64Image(B64Image),
}

/// The input to this module's ML engine -- a request for inference
#[derive(Deserialize)]
pub struct InferenceTask {
    pub data: InputData,
    pub inference_type: InferenceType,
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
        let output: IValue = no_grad(|| self.model.forward_is(&[img])).unwrap();

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

        let img = Image {
            image: buffer,
            height: Some(height as u32),
            width: Some(width as u32),
        };

        // Encode the buffer as base-64

        Ok(Inference::B64Image(img.into()))
    }

    /// Run inference on the loaded model given an `InferenceTask`
    pub fn run(&self, task: InferenceTask) -> Result<Inference> {
        match task.inference_type {
            InferenceType::ImageClassification { top_n } => match task.data {
                InputData::B64Image(image) => Ok(self.image_classification(image.into(), top_n)?),
                _ => Err(anyhow!(
                    "invalid input type for ImageClassification inference"
                )),
            },
            InferenceType::ImageToImage => match task.data {
                InputData::B64Image(image) => Ok(self.image_to_image(image.into())?),
                _ => Err(anyhow!("invalid input type for ImageToImage inference")),
            },
            _ => Err(anyhow!("that inference type is not currently supported")),
        }
    }
}

impl From<Vec<Class>> for rpc::Inference {
    fn from(classes: Vec<Class>) -> Self {
        rpc::Inference {
            text: None,
            image: None,
            classification: Some(rpc::Classes {
                classes: classes
                    .into_iter()
                    .map(|c| rpc::Class {
                        probability: c.probability,
                        label: c.label,
                    })
                    .collect(),
            }),
        }
    }
}

impl From<Inference> for rpc::Inference {
    fn from(inference: Inference) -> rpc::Inference {
        match inference {
            Inference::Text(text) => rpc::Inference {
                text: Some(text),
                image: None,
                classification: None,
            },
            Inference::Classification(c) => c.into(),
            Inference::B64Image(byte_str) => rpc::Inference {
                image: Some(rpc::B64Image {
                    //image: byte_str.as_bytes().to_vec(),
                    image: byte_str.image,
                    height: None,
                    width: None,
                }),
                text: None,
                classification: None,
            },
        }
    }
}

impl From<rpc::B64Image> for Image {
    fn from(image: rpc::B64Image) -> Image {
        Image {
            image: base64::decode(image.image).unwrap(),
            height: image.height,
            width: image.width,
        }
    }
}

impl From<rpc::InferenceTask> for InferenceTask {
    fn from(task: rpc::InferenceTask) -> InferenceTask {
        match task
            .inference_type
            .expect("must provide inference type")
            .r#type
        {
            // ImageClassification
            0 => InferenceTask {
                data: InputData::B64Image(
                    task.image
                        .expect("must provide image for ImageClassification inference")
                        .into(),
                ),
                inference_type: InferenceType::ImageClassification {
                    top_n: task.inference_type.unwrap().top_n.unwrap() as u16,
                },
            },
            // ImageToImage
            1 => InferenceTask {
                data: InputData::B64Image(
                    task.image
                        .expect("must provide image for ImageToImage inference")
                        .into(),
                ),
                inference_type: InferenceType::ImageToImage,
            },
            // TextToText
            2 => InferenceTask {
                data: InputData::Text(
                    task.text
                        .expect("must provide text for TextToText inference"),
                ),
                inference_type: InferenceType::TextToText,
            },
        }
    }
}

impl From<InferenceTask> for rpc::InferenceTask {
    fn from(task: InferenceTask) -> rpc::InferenceTask {
        match task.inference_type {
            InferenceType::ImageClassification { top_n } => rpc::InferenceTask {
                inference_type: Some(rpc::InferenceType {
                    r#type: 0,
                    top_n: Some(top_n as u32),
                }),
                image: match task.data {
                    InputData::B64Image(img) => Some(rpc::B64Image {
                        image: img.image,
                        height: img.height,
                        width: img.width,
                    }),
                    InputData::Text(_) => None,
                },
                text: None,
            },
            InferenceType::ImageToImage => {
                rpc::InferenceTask {
                    inference_type: Some(rpc::InferenceType {
                        r#type: 1,
                        top_n: None,
                    }),
                    image: match task.data {
                        InputData::B64Image(img) => Some(rpc::B64Image {
                            //image: img.image.into(),
                            image: img.image,
                            height: img.height,
                            width: img.width,
                        }),
                        InputData::Text(_) => None,
                    },
                    text: None,
                }
            }
            InferenceType::TextToText => unimplemented!(),
        }
    }
}

/*
impl From<rpc::Inference> for Inference {
    fn from(inference: rpc::Inference) -> Inference {
        Inference {
            data: inference.data,
            inference_type: inference.inference_type,
        }
    }
}
*/

impl From<rpc::Classes> for Vec<Class> {
    fn from(classes: rpc::Classes) -> Vec<Class> {
        classes
            .classes
            .into_iter()
            .map(|c| Class {
                probability: c.probability,
                label: c.label,
            })
            .collect()
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
