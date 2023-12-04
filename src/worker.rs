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

    test_image: Arc<Mutex<torch::InputData>>,
}

impl Worker {
    pub fn new(model_file: String, port: u16) -> Result<Self> {
        Ok(Worker {
            model: Arc::new(torch::TorchModel::new(model_file)?),
            port,
            test_image: Arc::new(Mutex::new(util::test::get_test_image())),
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
        //let mut s = self.status.lock().unwrap();
        //*s = WorkerStatus::Working;
        let res = self.model.run(input);
        //*s = WorkerStatus::Idle;
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

        //actix_web::rt::time::sleep(std::time::Duration::from_millis(4000)).await;

        // Run inference in a separate thread
        let res = model.run(image).unwrap();

        info!("worker successfully computed inference: {res:?}");
        Ok(Response::new(res.into()))
    }

    /*
        #[tracing::instrument]
        async fn image_inference(
            &self,
            _request: Request<ImageInput>,
        ) -> TResult<Response<ClassOutput>> {
            info!("worker got FAKE inference request");

            /*
            let output = actix_web::rt::task::spawn_blocking(move || ClassOutput {
                classes: vec![Classification {
                    probability: 0.4,
                    class_int: 3,
                    label: "book".to_string(),
                }],
                num_classes: 919,
            })
            .await
            .unwrap();
            */

            let img = self.test_image.clone();
            let model = self.model.clone();

            let output = tokio::task::spawn(async move {
                let i = img.lock().unwrap();
                let output = model.run(i.clone()).unwrap();
                output
            })
            .await
            .unwrap();

            info!("worker successfully FAKED inference: {output:?}");
            return Ok(Response::new(output.into()));
        }
    */

    // DEPRECATED
    async fn shutdown(&self, _request: Request<Empty>) -> TResult<Response<Empty>> {
        info!("worker got shutdown request");
        Ok(Response::new(Empty {}))
    }
}
