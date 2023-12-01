//! An inference worker listens for requests from the `Manager` and computes
//! model inference in an isolated environment

use crate::rpc::worker_server::{self, WorkerServer};
use crate::rpc::{ClassOutput, Empty, ImageInput, Status};
use crate::torch;
use anyhow::anyhow;
use anyhow::Result;
use log::{debug, info};
use serde::Serialize;
use std::sync::Mutex;
use tonic::transport::Server;
use tonic::{Request, Response};

type TResult<T> = Result<T, tonic::Status>;

/// The current status of a worker
#[derive(Debug, Clone, Serialize)]
pub enum WorkerStatus {
    /// Currently computing inference
    Working = 0,

    /// Not computing inference
    Idle,

    /// In the process of shutting down
    ShuttingDown,

    /// In an error state
    Error,
}

impl From<Status> for WorkerStatus {
    fn from(s: Status) -> Self {
        match s.status {
            0 => WorkerStatus::Working,
            1 => WorkerStatus::Idle,
            2 => WorkerStatus::ShuttingDown,
            3 => WorkerStatus::Error,
            _ => unreachable!(),
        }
    }
}

/// A worker runs as a separate process, spawned by the resource manager.
/// A worker runs an RPC server listening for requests to compute inference
/// on its own local copy of the model
#[derive(Debug)]
pub struct Worker {
    model: torch::TorchModel,
    status: Mutex<WorkerStatus>,
    port: u16,
}

impl Worker {
    pub fn new(model_file: String, port: u16) -> Result<Self> {
        Ok(Worker {
            model: torch::TorchModel::new(model_file)?,
            status: Mutex::new(WorkerStatus::Idle),
            port,
        })
    }

    /// Start listening for requests
    pub async fn start(self) -> Result<()> {
        info!(
            "starting new worker on port {} with model {:?}",
            self.port, self.model
        );
        let addr = format!("[::1]:{}", self.port).parse().unwrap();
        let svc = WorkerServer::new(self);
        Server::builder().add_service(svc).serve(addr).await?;
        Ok(())
    }

    /// Return the worker's status
    pub fn status(&self) -> WorkerStatus {
        (*self.status.lock().unwrap()).clone()
    }

    /// Run inference on the worker
    pub fn run(&self, input: torch::InputData) -> Result<torch::Inference> {
        self.model.run(input)
    }

    /// Shutdown the worker
    pub fn shutdown(&self) {}
}

#[tonic::async_trait]
impl worker_server::Worker for Worker {
    async fn image_inference(
        &self,
        _request: Request<ImageInput>,
    ) -> TResult<Response<ClassOutput>> {
        info!("got inference request");
        let image: torch::InputData = _request.into_inner().into();

        let output = self.run(image).unwrap();
        return Ok(Response::new(output.into()));
    }

    async fn get_status(&self, _request: Request<Empty>) -> TResult<Response<Status>> {
        info!("got status request");
        let status = self.status() as i32;
        Ok(Response::new(Status { status }))
    }

    // DEPRECATED
    async fn shutdown(&self, _request: Request<Empty>) -> TResult<Response<Empty>> {
        info!("got shutdown request");
        let mut g = self.status.lock().unwrap();
        *g = WorkerStatus::ShuttingDown;
        Ok(Response::new(Empty {}))
    }
}
