//! The manager (worker manager, resource allocator, allocator) is responsible for
//! interfacing with a set of workers. The manager starts and stops workers, and
//! forwards inference requests

use crate::config::{self, *};
use crate::rpc::{self, worker_client::WorkerClient};
use crate::torch::Inference;
use crate::torch::InputData;
use crate::util;
use crate::worker::WorkerStatus;
use anyhow::anyhow;
use anyhow::Result;
use log::{debug, error, info, warn};
use nix::{sys::signal, unistd};
use serde::Serialize;
use std::collections::HashMap;
use std::fs::File;
use std::process::{Command, Stdio};
use std::sync::Arc;
use std::sync::Mutex;
use std::sync::RwLock;
use std::{thread, time};
use tch::IndexOp;
use tokio_stream::StreamExt;
use tonic::transport::Channel;
use tonic::transport::Endpoint;
use tonic::{Request, Response};

type Connection = WorkerClient<Channel>;

/// A handle to a worker
#[derive(Clone, Serialize)]
pub struct Handle {
    pub port: u16,
    pub pid: u32,
    #[serde(skip_serializing)]
    pub conn: Connection,
}

/// The worker manager. Right now, assumes that all workers
/// are on the same host
pub struct Manager {
    /// Map from PID to `Handle`s of current workers
    //workers: Arc<RwLock<HashMap<u32, Handle>>>,
    //workers: Arc<Mutex<HashMap<u32, Handle>>>,
    workers: HashMap<u32, Handle>,

    /// The path to the TorchScript model file
    model_file: String,
}

impl Manager {
    pub fn new(model_file: String) -> Self {
        Manager {
            //workers: Arc::new(RwLock::new(HashMap::new())),
            //workers: Arc::new(Mutex::new(HashMap::new())),
            workers: HashMap::new(),
            model_file,
        }
    }

    /// Start a new worker process on the local machine and connect to it
    async fn start_new_worker(&mut self) -> Result<Handle> {
        //let mut workers = self.workers.lock().unwrap();
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

        let t = util::time();
        let out_name = format!("./logs/worker_{}_{}.out", port, t);
        let err_name = format!("./logs/worker_{}_{}.err", port, t);
        let out_log = File::create(out_name).expect("failed to open log");
        let err_log = File::create(err_name).expect("failed to open log");

        let pid = Command::new(WORKER_BINARY)
            .env("RUST_LOG", RUST_LOG)
            .args(args)
            .stdout(out_log)
            .stderr(err_log)
            .spawn()
            .unwrap()
            .id();

        info!("manager started new worker process {pid}");

        // Spin until connection succeeds, or times out
        let conn: Connection;
        let now = time::Instant::now();
        loop {
            // Connect to the new local worker
            match WorkerClient::connect(format!("http://[::1]:{port}")).await {
                Ok(c) => {
                    conn = c;
                    break;
                }
                Err(_) => {
                    if now.elapsed().as_millis() >= WORKER_TIMEOUT {
                        return Err(anyhow!("timeout connecting to new worker process"));
                    }
                }
            }
        }

        let handle = Handle { port, pid, conn };

        info!("manager successfully connected to new worker (port = {port}, pid = {pid})",);

        self.workers.insert(handle.pid, handle.clone());
        Ok(handle)
    }

    /// Get a handle to an idle worker, or start a new worker if no workers are currently idle
    async fn get_idle_worker(&mut self) -> Result<Handle> {
        // Loop through all registered workers and return the first idle one
        let mut handles = tokio_stream::iter(self.workers.values());
        while let Some(handle) = handles.next().await {
            let req = Request::new(rpc::Empty {});
            let res: WorkerStatus = handle
                .conn
                .clone()
                .get_status(req)
                .await?
                .into_inner()
                .into();
            match res {
                WorkerStatus::Idle => {
                    info!("found an idle worker: no need for a new worker");
                    return Ok(handle.clone());
                }
                _ => continue,
            }
        }

        // If there are no idle workers, make a new worker (guaranteed to be idle)
        info!("all workers are busy: attempting to start a new worker");
        self.start_new_worker().await
    }

    // ----- Interface ----- //

    /// Run inference on an idle worker
    //#[tracing::instrument]
    pub async fn run_inference(&mut self, input: InputData) -> Result<Inference> {
        // Find an idle worker
        let handle = self.get_idle_worker().await?;
        debug!("make unlazy inference handle: {handle:?}");
        // Send req
        let req = Request::new(input.into());

        // Reconnent to the RPC client
        let conn = WorkerClient::connect(format!("http://[::1]:{}", handle.port)).await?;

        // Send an RPC request for inference
        let res = conn.clone().image_inference(req).await?.into_inner().into();
        // Ok(Inference::Text("some inference".to_string()))
        Ok(res)
    }

    /// Get the statuses of all workers
    pub async fn all_status(&self) -> Result<HashMap<Handle, WorkerStatus>> {
        let mut map: HashMap<Handle, WorkerStatus> = HashMap::new();

        //let mut handles = tokio_stream::iter(workers.values().collect::<Vec<&Handle>>());
        let mut handles = tokio_stream::iter(self.workers.values());
        while let Some(handle) = handles.next().await {
            debug!("sending status request to worker pid = {}", handle.pid);
            let req = Request::new(rpc::Empty {});
            let res = handle.conn.clone().get_status(req).await?.into_inner();
            debug!("got status of worker: {res:?}");
            map.insert(handle.clone(), res.into());
        }

        Ok(map)
    }

    // idea: the Manager is being cloned in the data part of the actix framework

    /// Return all the workers, without their status
    pub fn all_workers(&self) -> Vec<Handle> {
        self.workers
            .values()
            .map(|w| Handle {
                pid: w.pid,
                port: w.port,
                conn: w.conn.clone(),
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
        let handle = self.get_idle_worker().await?;
        let pid = handle.pid;
        // Remove it from the worker store
        let handle = self.workers.remove(&pid).unwrap();
        // Kill the process
        signal::kill(unistd::Pid::from_raw(pid as i32), signal::Signal::SIGTERM).unwrap();
        Ok(handle)
    }
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
