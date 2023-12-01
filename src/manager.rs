//! The manager (worker manager, resource allocator, allocator) is responsible for
//! interfacing with a set of workers. The manager starts and stops workers, and
//! forwards inference requests

use crate::config::{self, WORKER_BINARY};
use crate::rpc::{self, worker_client::WorkerClient};
use crate::torch::Inference;
use crate::torch::InputData;
use crate::util;
use crate::worker::WorkerStatus;
use anyhow::anyhow;
use anyhow::Result;
use log::{debug, info};
use nix::{sys::signal, unistd};
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
    pub port: u16,
    pub pid: u32,
    pub conn: Option<WorkerClient<Channel>>,
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
#[derive(Debug)]
pub struct Manager {
    /// Map from PID to `Handle`s of current workers
    workers: HashMap<u32, Handle>,

    /// The path to the TorchScript model file
    model_file: String,
}

impl Manager {
    pub fn new(model_file: String) -> Self {
        Manager {
            workers: HashMap::new(),
            model_file,
        }
    }

    /// Start a new worker process on the local machine and connect to it
    async fn start_new_worker(&mut self) -> Result<()> {
        if self.workers.len() + 1 >= config::MAX_WORKERS {
            return Err(anyhow!(
                "maximum number of workers exceeded. cannot allocate any more",
            ));
        }
        // Find an open port
        let port = util::get_available_port().unwrap(); // Use ok_or here

        debug!("found free port {port}");

        // Start a new local worker process
        let command = format!("{} {}", port, self.model_file);
        let args = command.split(' ').map(|n| n.to_string());
        let pid = Command::new(WORKER_BINARY)
            .env("RUST_LOG", "debug")
            .args(args)
            .spawn()
            .unwrap()
            .id();

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

        self.workers.insert(handle.pid, handle);
        Ok(())
    }

    /// Find an idle worker
    async fn find_idle_worker(&self) -> Result<Handle> {
        let mut handles = tokio_stream::iter(self.workers.values());
        while let Some(handle) = handles.next().await {
            let conn = handle.conn.clone();
            let req = Request::new(rpc::Empty {});
            let res: WorkerStatus = conn.unwrap().get_status(req).await?.into_inner().into();
            match res {
                WorkerStatus::Idle => return Ok(handle.clone()),
                _ => continue,
            }
        }
        Err(anyhow!("no idle workers found"))
    }

    // ----- Interface ----- //

    /// Run inference on an idle worker
    pub async fn run_inference(&mut self, input: InputData) -> Result<Inference> {
        // Find an idle worker
        let handle = self.find_idle_worker().await?; // TODO: MAKE IT ADD A NEW ONE IF DOESN'T
                                                     // EXIST
        match handle.conn {
            Some(mut conn) => {
                let req = Request::new(input.into());
                let res = conn.image_inference(req).await?.into_inner().into();
                Ok(res)
            }
            None => todo!(), // Try to connect again
        }
    }

    /// Get the statuses of all workers
    pub async fn all_status(&mut self) -> Result<HashMap<Handle, WorkerStatus>> {
        let mut map: HashMap<Handle, WorkerStatus> = HashMap::new();
        if self.workers.len() == 0 {
            self.start_new_worker().await?;
        }

        let mut handles = tokio_stream::iter(self.workers.values().collect::<Vec<&Handle>>());
        while let Some(handle) = handles.next().await {
            debug!("getting status of worker pid {}", handle.pid);
            let conn = handle.conn.clone();
            let req = Request::new(rpc::Empty {});
            let res = conn.unwrap().get_status(req).await?.into_inner();
            map.insert(handle.clone(), res.into());
        }

        Ok(map)
    }

    /// Start a new worker process on the local machine and connect to it
    pub async fn start_new_workers(&mut self, n: u16) -> Result<()> {
        let mut stream = tokio_stream::iter(0..n);
        while let Some(_) = stream.next().await {
            self.start_new_worker().await?;
        }
        Ok(())
    }

    /// Kill an idle worker and return a partial handle to it
    pub async fn kill_worker(&mut self) -> Result<Handle> {
        // Find an idle worker
        let pid = self.find_idle_worker().await?.pid;
        // Remove it from the worker store
        let handle = self.workers.remove(&pid).unwrap();
        // Kill the process
        signal::kill(unistd::Pid::from_raw(pid as i32), signal::Signal::SIGTERM).unwrap();
        Ok(handle)
    }
}
