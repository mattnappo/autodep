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
use std::{fmt::Debug, io::Cursor};
use tch::vision::imagenet;
use tch::{IValue, Kind};

use image::GenericImageView;
use tch::{nn, no_grad, vision, Device, Tensor};

/// An in-memory representation of an image (not base 64). Can be the input or output of a model
#[derive(Debug, Serialize, Clone, Deserialize)]
pub struct Image {
    pub(crate) image: Vec<u8>,
    pub(crate) height: Option<u32>,
    pub(crate) width: Option<u32>,
}

/// A base 64 image
#[derive(Serialize, Deserialize)]
pub struct B64Image {
    pub image: String,
    pub height: Option<u32>,
    pub width: Option<u32>,
}

impl Debug for B64Image {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "B64Image {{ img: <data>, height: {:?}, width: {:?} }}",
            self.height, self.width
        )
    }
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

/// The type of inference to compute
#[derive(Serialize, Deserialize, Debug, Clone)]
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
#[derive(Deserialize, Debug, Serialize)]
pub enum InputData {
    Text(String),
    //Image(Image),
    B64Image(B64Image),
}

/// The input to this module's ML engine -- a request for inference
#[derive(Debug, Serialize, Deserialize)]
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
        // Load image and convert to float tensor
        let img = tch::vision::image::load_from_memory(&image.image)?;
        let img = img.to_kind(tch::Kind::Float) / 255.;

        // Add a batch dimension
        let img = img.unsqueeze(0);
        let img = IValue::Tensor(img);

        // Run the model on the image
        let output = no_grad(|| self.model.forward_is(&[img]))?;
        let (_, output) = match output {
            IValue::GenericDict(tensors) => tensors.into_iter().find(|(label, _)| match label {
                IValue::String(s) => s == "out",
                _ => false,
            }),
            _ => {
                return Err(anyhow!(
                    "image-to-image inference failed on the forward step"
                ))
            }
        }
        .unwrap();

        // Extract the tensor
        let output_predictions = match output {
            IValue::Tensor(t) => t.squeeze().argmax(0, false),
            _ => {
                return Err(anyhow!(
                    "image-to-image inference failed to return a tensor"
                ))
            }
        };

        // Create the palette and colors
        let palette = Tensor::from_slice(&[2i64.pow(25) - 1, 2i64.pow(15) - 1, 2i64.pow(21) - 1]);
        let colors: Tensor =
            Tensor::from_slice(&(0..21).collect::<Vec<_>>()[..]).unsqueeze(-1) * palette;
        let colors = colors.remainder(255).to_kind(Kind::Uint8);

        let (width, height) = (output_predictions.size()[1], output_predictions.size()[0]);

        // Wrtie tensor to vector
        let mut output_data = vec![];
        let mut i = output_predictions.view([-1]).iter::<i64>().unwrap();
        while let Some(class_index) = i.next() {
            let color = colors.get(class_index as i64);
            let mut dst = vec![0; 3];
            color.copy_data(&mut dst, 3);
            output_data.extend_from_slice(&dst);
        }

        // Convert tensor to rgb image
        let output_image: ImageBuffer<Rgb<u8>, _> =
            ImageBuffer::from_raw(width as u32, height as u32, output_data).unwrap();

        // Write to b64
        let mut image_data: Vec<u8> = Vec::new();
        output_image
            .write_to(&mut Cursor::new(&mut image_data), ImageOutputFormat::Png)
            .unwrap();
        let b64img = base64::encode(image_data);
        Ok(Inference::B64Image(B64Image {
            image: b64img,
            height: Some(height as u32),
            width: Some(width as u32),
        }))
    }

    pub fn text_to_text(&self, task: InferenceTask) -> Result<Inference> {
        let device = Device::Cpu;

        // Create example input tensor
        let input_ids = Tensor::from_slice(&[0, 1, 2, 3, 4]).to_device(device);
        let attention_mask = Tensor::from_slice(&[1, 1, 1, 1, 1]).to_device(device);
        let token_type_ids = Tensor::from_slice(&[0, 0, 0, 0, 0]).to_device(device);

        // Run the model with no_grad to prevent tracking history
        let result = no_grad(|| {
            self.model
                .forward_ts(&[
                    input_ids.unsqueeze(0),
                    attention_mask.unsqueeze(0),
                    token_type_ids.unsqueeze(0),
                ])
                .unwrap()
        });

        println!("{:?}", result);

        todo!()
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
        let ty = task.inference_type.clone();
        match ty.expect("must provide inference type").r#type {
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
            _ => unreachable!(),
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
            InferenceType::TextToText => {
                unimplemented!("inference type TextToText not implemented yet")
            }
        }
    }
}

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

    #[test]
    fn test_deeplabv3() {
        let loader = TorchModel::new("models/new_deeplab_v3.pt".into()).unwrap();

        let task = InferenceTask {
            //data: test::load_image_from_disk("images/seg3.jpg".into()),
            data: test::load_image_from_disk("images/seg1.jpg".into()),
            inference_type: InferenceType::ImageToImage,
        };
        let outputs = loader.run(task).unwrap();
        println!("outputs: {outputs:#?}");
    }

    #[test]
    fn test_faster_rcnn() {
        let loader = TorchModel::new("models/faster_rcnn.pt".into()).unwrap();

        let task = InferenceTask {
            data: test::load_image_from_disk("images/seg1.png".into()),
            inference_type: InferenceType::ImageToImage,
        };
        let outputs = loader.run(task).unwrap();
        println!("outputs: {outputs:#?}");
    }

    use serde_json;

    #[test]
    fn test_ser() {
        let task = InferenceTask {
            data: test::load_image_from_disk("images/seg1.png".into()),
            inference_type: InferenceType::ImageClassification { top_n: 3 },
        };

        println!("{}", serde_json::to_string(&task).unwrap());
    }

    #[test]
    fn test_bert() {
        let loader = TorchModel::new("models/bert.pt".into()).unwrap();
        let task = InferenceTask {
            data: InputData::Text("hello <mask> is a common beginner programm".into()),
            inference_type: InferenceType::TextToText,
        };
        let outputs = loader.run(task).unwrap();
        println!("outputs: {outputs:#?}");
    }
}
