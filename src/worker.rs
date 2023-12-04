//! An inference worker listens for requests from the `Manager` and computes
//! model inference in an isolated environment

use crate::config::*;
use crate::rpc::worker_server::{self, WorkerServer};
use crate::rpc::{ClassOutput, Classification, Empty, ImageInput};
use crate::torch::{self, Class};
use crate::util;
use anyhow::anyhow;
use anyhow::Result;
use serde::Serialize;
use std::io::Write;
use std::sync::{Arc, Mutex};
use std::time::Duration;
use tonic::transport::Server;
use tonic::{Request, Response};
use tracing::*;

type TResult<T> = Result<T, tonic::Status>;

/// The current status of a worker
#[derive(Debug, Clone, Serialize, PartialEq)]
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

/// A worker runs as a separate process, spawned by the resource manager.
/// A worker runs an RPC server listening for requests to compute inference
/// on its own local copy of the model
#[derive(Debug)]
pub struct Worker {
    model: Arc<torch::TorchModel>,
    port: u16,
}

impl Worker {
    pub fn new(model_file: String, port: u16) -> Result<Self> {
        Ok(Worker {
            model: Arc::new(torch::TorchModel::new(model_file)?),
            port,
        })
    }

    /// Start listening for requests
    #[tracing::instrument]
    pub async fn start(self) -> Result<()> {
        info!(
            "starting new worker on port {} with model {:?}",
            self.port, self.model
        );
        let addr = format!("[::1]:{}", self.port).parse().unwrap();
        let svc = WorkerServer::new(self);
        Server::builder()
            .tcp_keepalive(Some(Duration::from_millis(1000)))
            .concurrency_limit_per_connection(32)
            .add_service(svc)
            .serve(addr)
            .await?;
        Ok(())
    }

    /// Run inference on the worker
    #[tracing::instrument]
    pub fn run(&self, input: torch::InputData) -> Result<torch::Inference> {
        self.model.run(input)
    }
}

#[tonic::async_trait]
impl worker_server::Worker for Worker {
    async fn image_inference(
        &self,
        _request: Request<ImageInput>,
    ) -> TResult<Response<ClassOutput>> {
        info!("worker got inference request");
        // Parse input request
        let image: torch::InputData = _request.into_inner().into();

        // Run model inference
        let model = self.model.clone();
        let res = model.run(image).unwrap();
        //let res = self.run(image).unwrap();

        info!("worker successfully computed inference: {res:?}");
        Ok(Response::new(res.into()))
    }
}
