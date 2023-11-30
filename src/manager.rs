//! The manager (worker manager, resource allocator, allocator) is responsible for
//! interfacing with a set of workers. The manager starts and stops workers, and
//! forwards inference requests

use crate::config::WORKER_BINARY;
use crate::rpc::{self, worker_client::WorkerClient};
use crate::util;
use crate::worker::WorkerStatus;
use anyhow::Result;
use log::{debug, info};
use std::collections::HashMap;
use std::process::Command;
use std::{thread, time};
use tch::IndexOp;
use tokio_stream::StreamExt;
use tonic::transport::Channel;
use tonic::{Request, Response};

/// A handle to a worker
#[derive(Debug, Clone)]
pub struct Handle {
    port: u16,
    pid: u32,
    conn: Option<WorkerClient<Channel>>,
}

impl std::hash::Hash for Handle {
    fn hash<H>(&self, state: &mut H)
    where
        H: std::hash::Hasher,
    {
        // Skip hashing the conn
        (&self.port, &self.pid).hash(state);
    }
}
impl PartialEq for Handle {
    fn eq(&self, other: &Self) -> bool {
        self.port == other.port && self.pid == other.pid
    }
}
impl Eq for Handle {}

/// The worker manager. Right now, assumes that all workers
/// are on the same host
pub struct Manager {
    /// The ports of the current workers
    workers: Vec<Handle>,

    /// The path to the TorchScript model file
    model_file: String,
}

impl Manager {
    pub fn new(model_file: String) -> Self {
        Manager {
            workers: vec![],
            model_file,
        }
    }

    /// Start a new worker process on the local machine and connect to it
    pub async fn start_new_worker(&mut self) -> Result<()> {
        // Find an open port
        let port = util::get_available_port().unwrap(); // Use ok_or here

        debug!("found free port {port}");

        // Start a new local worker process
        let command = format!("{} {}", port, self.model_file);
        let args = command.split(' ').map(|n| n.to_string());
        let pid = Command::new(WORKER_BINARY).args(args).spawn().unwrap().id();

        // Wait for the local process to start
        std::thread::sleep(time::Duration::from_millis(1000));

        // Connect to the new local worker
        let client = WorkerClient::connect(format!("http://[::1]:{port}")).await?;

        let handle = Handle {
            port,
            pid,
            conn: Some(client),
        };

        info!(
            "manager started a new worker on (port = {}, pid = {})",
            port, pid
        );

        self.workers.push(handle);
        Ok(())
    }

    /// Start a new worker process on the local machine and connect to it
    pub async fn start_new_workers(&mut self, n: u16) -> Result<()> {
        let mut stream = tokio_stream::iter(0..n);
        while let Some(_) = stream.next().await {
            self.start_new_worker().await?;
        }
        Ok(())
    }

    /// Get the statuses of all workers
    pub async fn all_status(&self) -> Result<HashMap<Handle, WorkerStatus>> {
        //let conns = self.workers.iter().map(|w| w.conn).filter(|w| w.is_some());
        let mut map: HashMap<Handle, WorkerStatus> = HashMap::new();

        let mut stream = tokio_stream::iter(&self.workers);
        while let Some(handle) = stream.next().await {
            debug!("getting status of worker pid {}", handle.pid);
            let conn = handle.conn.clone();
            let req = Request::new(rpc::Empty {});
            let res = conn.unwrap().get_status(req).await?.into_inner();
            map.insert(handle.clone(), res.into());
            //debug!("map: {map:#?}");
            //debug!("workers internal: {:#?}", self.workers);
            if self.workers.len() == map.len() {
                break;
            }
            debug!("still polling");
        }

        Ok(map)
    }

    //pub(crate) async fn kill_worker(&mut self) {}

    /// A testing function -- ignore
    pub async fn test(&self, port: u16) -> Result<()> {
        let mut client = WorkerClient::connect(format!("http://[::1]:{port}")).await?;

        let req = Request::new(rpc::Empty {});
        let res = client.get_status(req).await?;

        println!("RESPONSE = {:?}", res);

        Ok(())
    }
}
