//! The manager (worker manager, resource allocator, allocator) is responsible for
//! interfacing with a set of workers. The manager starts and stops workers, and
//! forwards inference requests

use crate::rpc::{self, worker_client::WorkerClient};
use crate::util;
use anyhow::Result;
use tonic::Request;

/// The worker manager. Right now, assumes that all workers
/// are on the same host
pub struct Manager {
    /// The ports of the current workers
    workers: Vec<u16>,
}

impl Manager {
    pub fn new() -> Self {
        Manager { workers: vec![] }
    }

    /// Start a new worker process on the local machine, and connect to it
    pub(crate) async fn alloc_worker() -> Result<()> {
        // Find an open port
        let port = util::get_available_port().unwrap(); // Use ok_or here

        // Start a new local worker process

        // Connect to the worker
        let mut client = WorkerClient::connect(format!("http://[::1]:{port}")).await?;

        let req = Request::new(rpc::Empty {});
        let res = client.get_status(req).await?;

        println!("RESPONSE = {:?}", res);

        Ok(())
    }

    pub async fn test(&self, port: u16) -> Result<()> {
        let mut client = WorkerClient::connect(format!("http://[::1]:{port}")).await?;

        let req = Request::new(rpc::Empty {});
        let res = client.get_status(req).await?;

        println!("RESPONSE = {:?}", res);

        Ok(())
    }
}
