/// Code for loading and running (trained) PyTorch models
use anyhow::Result;

/// An image: input or output of a model
struct Image {
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
            filename,
            model: tch::CModule::load(filename)?,
        })
    }

    /// Run inference on the model
    pub fn run(&self, input: InputData) -> Result<()> {
        match input {
            InputData::Text(_) => todo!(),
            InputData::Image(path) => {
                let image = imagenet::load_image_and_resize(path)?;
                let output = self.model.forward_ts(&[image.unsqueeze(0)])?.softmax(-1);
                for (probability, class) in imagenet::top(&output, 5).iter() {
                    println!("{:50} {:5.2}%", class, 100.0 * probability)
                }
            }
        }
        Ok(())
    }
}

#[cgf(tests)]
mod tests {
    #[test]
    fn test_load() {
        println!("hello");
        let loader = TorchLoader("models/resnet18.pt").unwrap();
        loader.run(InputData::Image("images/cat.png")).unwrap();
    }
}
