use crate::manager::Handle;
use crate::worker::WorkerStatus;
use serde::ser::{Serialize, SerializeMap, Serializer};
use serde::Deserialize;
use std::collections::HashMap;

pub struct AllStatusResponse(pub HashMap<Handle, WorkerStatus>);

impl Serialize for AllStatusResponse {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        let mut map = serializer.serialize_map(Some(self.0.len()))?;
        for (k, v) in &self.0 {
            map.serialize_entry(&format!("{k:?}"), &format!("{v:?}"))?;
        }
        map.end()
    }
}

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
