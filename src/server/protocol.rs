use serde::Deserialize;

/// An in-memory representation of an image, encoded as base 64
#[derive(Deserialize)]
pub struct B64Image {
    pub image: String,
    pub height: Option<u32>,
    pub width: Option<u32>,
}

#[derive(Deserialize)]
pub enum InferenceRequest {
    Image(B64Image),
    Text(String),
}
