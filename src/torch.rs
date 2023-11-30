//! Code for loading and running (trained) PyTorch models

use anyhow::Result;
use tch::vision::imagenet;

const TOP_N: i64 = 5;

/// An in-memory representation of an image. Can be the input or output of a model
#[derive(Debug)]
pub struct Image {
    image: Vec<u8>,
    height: Option<u32>,
    width: Option<u32>,
}

/// A class prediction outputted by a classifier model
#[derive(Debug)]
pub struct Class {
    probability: Option<f64>,
    class: Option<i32>,
    label: Option<String>,
}

/// Input data that inference can be computed on
#[derive(Debug)]
pub enum InputData {
    Text(String),
    //Image(Image),
    Image(Image),
}

/// Data that a model inference could return
#[derive(Debug)]
pub enum OutputData {
    Text(String),
    Classes(Vec<Class>),
    Image(Image),
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
    pub fn run(&self, input: InputData) -> Result<OutputData> {
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
                Ok(OutputData::Classes(classes))
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use std::io::Read;

    #[test]
    fn test_run() {
        let loader = TorchModel::new("models/resnet18.pt".into()).unwrap();

        let mut file = fs::File::open("images/cat.png").unwrap();
        let mut image: Vec<u8> = vec![];
        file.read_to_end(&mut image).unwrap();

        let outputs = loader
            .run(InputData::Image(Image {
                image,
                height: None,
                width: None,
            }))
            .unwrap();

        println!("outputs: {outputs:#?}");
    }
}
