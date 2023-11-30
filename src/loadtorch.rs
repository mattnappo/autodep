/// Code for loading and running (trained) PyTorch models
use anyhow::Result;
use tch::vision::imagenet;

/// An image: input or output of a model
pub struct Image {
    image: Vec<u8>,
    height: u32,
    width: u32,
}

/// Input data that inference can be computed on
pub enum InputData {
    Text(String),
    //Image(Image),
    Image(String), // The path to the image
}

/// Data that a model inference could return
pub enum OutputData {
    Text(String),
    Class {
        probability: Option<f32>,
        class: i32,
        label: Option<String>,
    },
    Image(Image),
}

/// Load and run a TorchScript file
pub struct TorchLoader {
    /// TorchScript filename
    filename: String,

    /// The loaded torch model
    model: tch::jit::CModule,
    // ModelType (classifier, etc...)
}

impl TorchLoader {
    pub fn new(filename: String) -> Result<Self> {
        Ok(TorchLoader {
            filename: filename.clone(),
            model: tch::CModule::load(filename)?,
        })
    }

    /// Run inference on the model
    pub fn run(&self, input: InputData) -> Result<()> {
        match input {
            InputData::Text(_) => todo!(),
            InputData::Image(path) => {
                let image = imagenet::load_image_and_resize(path, 224, 224)?;
                let output = self
                    .model
                    .forward_ts(&[image.unsqueeze(0)])?
                    .softmax(-1, Some(tch::kind::Kind::Float));
                for (probability, class) in imagenet::top(&output, 5).iter() {
                    println!("{:50} {:5.2}%", class, 100.0 * probability)
                }
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_load() {
        println!("hello");
        let loader = TorchLoader::new("models/resnet18.pt".into()).unwrap();
        loader
            .run(InputData::Image("images/cat.png".into()))
            .unwrap();
    }
}
