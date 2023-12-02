//! Code for loading and running (trained) PyTorch models

use crate::config::TOP_N;
use crate::rpc;
use anyhow::Result;
use serde::Serialize;
use tch::vision::imagenet;

/// An in-memory representation of an image. Can be the input or output of a model
#[derive(Debug, Serialize)]
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
    Classes(Vec<Class>),
    Image(Image),
}

/// Input data that inference can be computed on
#[derive(Debug)]
pub enum InputData {
    Text(String),
    //Image(Image),
    Image(Image),
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
        Inference::Classes(
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
            Inference::Classes(c) => {
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

/// Load and run a TorchScript file
#[derive(Debug)]
pub struct TorchModel {
    /// TorchScript filename
    filename: String,

    /// The loaded torch model
    model: tch::jit::CModule,
    // ModelType (classifier, etc...)
}

impl TorchModel {
    pub fn new(filename: String) -> Result<Self> {
        Ok(TorchModel {
            filename: filename.clone(),
            model: tch::CModule::load(filename)?,
        })
    }

    /// Run inference on the loaded model
    /// Right now this function is not very general, and is somewhat hardcoded
    /// for imagenet. Will circle back later
    pub fn run(&self, input: InputData) -> Result<Inference> {
        //std::thread::sleep(std::time::Duration::from_millis(5000)); -- not working
        match input {
            InputData::Text(_) => todo!(),
            InputData::Image(image) => {
                let image = imagenet::load_image_from_memory(&image.image)?;
                let output = self
                    .model
                    .forward_ts(&[image.unsqueeze(0)])?
                    .softmax(-1, Some(tch::kind::Kind::Float));
                let classes = imagenet::top(&output, TOP_N)
                    .iter()
                    .map(|(p, l)| Class {
                        probability: Some(*p),
                        class: None,
                        label: Some(l.into()),
                    })
                    .collect();
                Ok(Inference::Classes(classes))
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::util::test;

    #[test]
    fn test_run() {
        let loader = TorchModel::new("models/resnet18.pt".into()).unwrap();
        let img = test::get_test_image();
        let outputs = loader.run(img).unwrap();
        println!("outputs: {outputs:#?}");
    }
}
