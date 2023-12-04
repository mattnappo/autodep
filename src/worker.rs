//! An inference worker listens for requests from the `Manager` and computes
//! model inference in an isolated environment

use crate::rpc;
use crate::rpc::worker_server::{self, WorkerServer};
use crate::torch;

use serde::Serialize;
use std::sync::Arc;
use std::time::Duration;
use tonic::transport::Server;
use tonic::{Request, Response};
use tracing::*;

type Result<T> = std::result::Result<T, tonic::Status>;

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
    pub fn new(model_file: String, port: u16) -> anyhow::Result<Self> {
        Ok(Worker {
            model: Arc::new(torch::TorchModel::new(model_file)?),
            port,
        })
    }

    /// Start listening for requests
    #[tracing::instrument]
    pub async fn start(self) -> anyhow::Result<()> {
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

    /*
    /// Run inference on the worker
    #[tracing::instrument]
    pub fn run(&self, input: torch::InputData) -> anyhow::Result<torch::Inference> {
        self.model.run(input)
    }
    */
}

#[tonic::async_trait]
impl worker_server::Worker for Worker {
    /// Handle requests for inference, and compute inference on this worker
    async fn compute_inference(
        &self,
        request: Request<rpc::InferenceTask>,
    ) -> Result<Response<rpc::Inference>> {
        info!("worker got inference request");
        // Parse input request
        let task: torch::InferenceTask = request.into_inner().into();
        debug!("task: {:?}", task);

        // Run model inference
        let model = self.model.clone();
        let res = model.run(task).unwrap();

        info!("worker successfully computed inference: {res:?}");
        Ok(Response::new(res.into()))
    }
}
