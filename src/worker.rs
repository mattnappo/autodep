//! An inference worker listens for requests from the `Manager` and computes
//! model inference in an isolated environment

use crate::rpc::worker_server::{self, WorkerServer};
use crate::rpc::{ClassOutput, Empty, ImageInput, Status};
use crate::torch;
use anyhow::anyhow;
use anyhow::Result;
use log::{debug, error, info, warn};
use serde::Serialize;
use std::io::Write;
use std::sync::{Arc, Mutex};
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
            3 | _ => WorkerStatus::Error,
            //_ => unreachable!(),
        }
    }
}

/// A worker runs as a separate process, spawned by the resource manager.
/// A worker runs an RPC server listening for requests to compute inference
/// on its own local copy of the model
#[derive(Debug)]
pub struct Worker {
    model: Arc<torch::TorchModel>,
    status: Arc<Mutex<WorkerStatus>>,
    port: u16,
}

impl Worker {
    pub fn new(model_file: String, port: u16) -> Result<Self> {
        Ok(Worker {
            model: Arc::new(torch::TorchModel::new(model_file)?),
            status: Arc::new(Mutex::new(WorkerStatus::Idle)),
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

    /// Run inference on the worker
    pub fn run(&self, input: torch::InputData) -> Result<torch::Inference> {
        let mut s = self.status.lock().unwrap();
        *s = WorkerStatus::Working;
        let res = self.model.run(input);
        *s = WorkerStatus::Idle;
        res
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
        info!("worker got inference request");
        let image: torch::InputData = _request.into_inner().into();

        let model = self.model.clone();
        let status = self.status.clone();

        let output = actix_web::rt::task::spawn_blocking(move || {
            let mut s = status.lock().unwrap();
            *s = WorkerStatus::Working;
            let res = model.run(image);
            *s = WorkerStatus::Idle;
            res
        })
        .await
        .unwrap();

        info!("worker successfully computed inference: {output:?}");
        return Ok(Response::new(output.unwrap().into()));
    }

    async fn get_status(&self, _request: Request<Empty>) -> TResult<Response<Status>> {
        info!("worker got status request");
        let status = self.status.lock().unwrap();
        Ok(Response::new(Status {
            status: status.clone() as i32,
        }))
    }

    // DEPRECATED
    async fn shutdown(&self, _request: Request<Empty>) -> TResult<Response<Empty>> {
        info!("worker got shutdown request");
        let mut g = self.status.lock().unwrap();
        *g = WorkerStatus::ShuttingDown;
        Ok(Response::new(Empty {}))
    }
}
