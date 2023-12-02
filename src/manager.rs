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
use log::{debug, info, warn};
use nix::{sys::signal, unistd};
use serde::Serialize;
use std::collections::HashMap;
use std::process::Command;
use std::sync::Arc;
use std::sync::RwLock;
use std::{thread, time};
use tch::IndexOp;
use tokio_stream::StreamExt;
use tonic::transport::Channel;
use tonic::{Request, Response};

/// A handle to a worker
#[derive(Clone, Serialize)]
pub struct Handle {
    pub port: u16,
    pub pid: u32,
    #[serde(skip_serializing)]
    pub conn: Option<WorkerClient<Channel>>,
}

impl std::fmt::Debug for Handle {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Handle {{ port: {}, pid: {} }}", self.port, self.pid)
    }
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
    /// Map from PID to `Handle`s of current workers
    workers: Arc<RwLock<HashMap<u32, Handle>>>,

    /// The path to the TorchScript model file
    model_file: String,
}

impl Manager {
    pub fn new(model_file: String) -> Self {
        Manager {
            workers: Arc::new(RwLock::new(HashMap::new())),
            model_file,
        }
    }

    /// Start a new worker process on the local machine and connect to it
    async fn start_new_worker(&mut self) -> Result<()> {
        let mut workers = (*self.workers).write().unwrap();
        if workers.len() + 1 >= config::MAX_WORKERS {
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

        warn!(
            "manager started a new worker on (port = {}, pid = {})",
            port, pid
        );

        //workers.insert(handle.pid, handle);
        workers.insert(handle.pid, handle);
        warn!("upon add: {workers:?}");
        Ok(())
    }

    /// Find an idle worker
    async fn find_idle_worker(&self) -> Result<Option<Handle>> {
        let workers = (*self.workers).read().unwrap();
        let mut handles = tokio_stream::iter(workers.values());
        while let Some(handle) = handles.next().await {
            let conn = handle.conn.clone();
            let req = Request::new(rpc::Empty {});
            let res: WorkerStatus = conn.unwrap().get_status(req).await?.into_inner().into();
            match res {
                WorkerStatus::Idle => return Ok(Some(handle.clone())),
                _ => continue,
            }
        }
        Ok(None)
    }

    // ----- Interface ----- //

    /// Run inference on an idle worker
    pub async fn run_inference(&mut self, input: InputData) -> Result<Inference> {
        // Find an idle worker
        if let Some(handle) = self.find_idle_worker().await? {
            match handle.conn {
                Some(mut conn) => {
                    let req = Request::new(input.into());
                    let res = conn.image_inference(req).await?.into_inner().into();
                    Ok(res)
                }
                None => todo!(), // Try to connect again
            }
        } else {
            self.start_new_worker().await?;

            // TODO: Terrible code
            if let Some(handle) = self.find_idle_worker().await? {
                match handle.conn {
                    Some(mut conn) => {
                        let req = Request::new(input.into());
                        let res = conn.image_inference(req).await?.into_inner().into();
                        Ok(res)
                    }
                    None => todo!(), // Try to connect again
                }
            } else {
                Err(anyhow!("couldn't find an available worker"))
            }
        }
    }

    /// Get the statuses of all workers
    pub async fn all_status(&mut self) -> Result<HashMap<Handle, WorkerStatus>> {
        let mut map: HashMap<Handle, WorkerStatus> = HashMap::new();

        let workers = self.workers.read().unwrap();
        //let mut handles = tokio_stream::iter(workers.values().collect::<Vec<&Handle>>());
        let mut handles = tokio_stream::iter(workers.values());
        while let Some(handle) = handles.next().await {
            debug!("getting status of worker pid {}", handle.pid);
            let conn = handle.conn.clone();
            let req = Request::new(rpc::Empty {});
            let res = conn.unwrap().get_status(req).await?.into_inner();
            map.insert(handle.clone(), res.into());
        }

        Ok(map)
    }

    /// Return all the workers, without their status
    pub fn all_workers(&mut self) -> Vec<Handle> {
        let workers = self.workers.read().unwrap();
        warn!("WORKERS INTERNAL: {:#?}", workers.values());
        workers
            .values()
            .map(|w| Handle {
                pid: w.pid,
                port: w.port,
                conn: None,
            })
            .collect()
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
        match self.find_idle_worker().await? {
            // If that worker exists, kill it
            Some(h) => {
                let pid = h.pid;
                // Remove it from the worker store
                let mut workers = self.workers.write().unwrap();
                let handle = workers.remove(&pid).unwrap();
                // Kill the process
                signal::kill(unistd::Pid::from_raw(pid as i32), signal::Signal::SIGTERM).unwrap();
                Ok(handle)
            }
            // If not, return err
            None => Err(anyhow!("no idle workers to kill")),
        }
    }
}
