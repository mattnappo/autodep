/// An inference worker listens for requests and computes model inference in
/// an isolated environment
use crate::torch;
use anyhow::Result;
use rpc::worker_server;
use rpc::{ClassOutput, Empty, ImageInput, Status};
use std::sync::Mutex;

use std::pin::Pin;
use std::sync::Arc;
use tokio::sync::mpsc;
use tonic::{Request, Response};

type TResult<T> = Result<T, tonic::Status>;

pub mod rpc {
    tonic::include_proto!("worker");
}

/// The current status of a worker
#[derive(Debug, Clone)]
pub enum WorkerStatus {
    /// Currently computing inference
    Working,

    /// Not computing inference
    Idle,

    /// In the process of shutting down
    ShuttingDown,

    /// In an error state
    Error,
}

/// A worker runs as a separate process, spawned by the resource manager.
/// A worker runs an RPC server listening for requests to compute inference
/// on its own local copy of the model
#[derive(Debug)]
pub struct Worker {
    model: torch::TorchModel,
    status: Mutex<WorkerStatus>,
    // TODO stats
}

impl Worker {
    pub fn new(model_filename: String) -> Result<Self> {
        Ok(Worker {
            model: torch::TorchModel::new(model_filename)?,
            status: Mutex::new(WorkerStatus::Idle),
        })
    }

    /// Start listening for requests
    pub fn start(&self) {}

    /// Return the worker's status
    pub fn status(&self) -> WorkerStatus {
        (*self.status.lock().unwrap()).clone()
    }

    /// Run inference on the worker
    pub fn run(&self, input: torch::InputData) -> Result<torch::OutputData> {
        self.model.run(input)
    }

    /// Shutdown the worker
    pub fn shutdown(&self) {}
}

#[tonic::async_trait]
impl worker_server::Worker for Worker {
    async fn get_status(&self, _request: Request<Empty>) -> TResult<Response<Status>> {
        unimplemented!()
    }

    async fn image_inference(
        &self,
        _request: Request<ImageInput>,
    ) -> TResult<Response<ClassOutput>> {
        unimplemented!()
    }

    async fn shutdown(&self, _request: Request<Empty>) -> TResult<Response<Empty>> {
        unimplemented!()
    }
}
