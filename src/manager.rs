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
use nix::{sys::signal, unistd};
use serde::ser::{Serialize, Serializer};
use serde::Serialize as DeriveSer;
use std::collections::HashMap;
use std::fs::File;
use std::process::{Command, Stdio};
use std::sync::Arc;
use std::sync::Mutex;
use std::sync::RwLock;
use std::{thread, time};
use tch::IndexOp;
use tokio::fs;
use tokio_stream::StreamExt;
use tonic::transport::Channel;
use tonic::transport::Endpoint;
use tonic::transport::Uri;
use tonic::{Request, Response};
use tracing::*;

//type Connection = WorkerClient<Channel>;

/*
impl Clone for Connection {
    fn clone(&self) -> Self {
        self.clone()
    }
}
*/

/// A handle to a worker
#[derive(Clone)]
pub struct Handle {
    pub port: u16,
    pub pid: u32,
    pub channel: Channel,
}

impl Serialize for Handle {
    fn serialize<S>(&self, serializer: S) -> std::prelude::v1::Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        let s = format!("{:?}", self);
        serializer.serialize_str(&s)
    }
}

/// The worker manager. Right now, assumes that all workers
/// are on the same host
#[derive(Debug)]
pub struct Manager {
    /// Map from PID to `Handle`s of current workers
    //workers: Arc<RwLock<HashMap<u32, Handle>>>,
    //workers: Arc<Mutex<HashMap<u32, Handle>>>,
    pub workers: HashMap<u32, (Handle, WorkerStatus)>,

    /// The path to the TorchScript model file
    model_file: String,
}

impl Manager {
    // should these be atomic?
    pub fn mark_working(&mut self, pid: u32) {
        let (h, _) = self.workers.get(&pid).unwrap();
        self.workers.insert(pid, (h.clone(), WorkerStatus::Working));
    }
    pub fn mark_idle(&mut self, pid: u32) {
        let (h, _) = self.workers.get(&pid).unwrap();
        self.workers.insert(pid, (h.clone(), WorkerStatus::Idle));
    }

    pub async fn new(model_file: String) -> Result<Self> {
        let mut m = Manager {
            //workers: Arc::new(RwLock::new(HashMap::new())),
            //workers: Arc::new(Mutex::new(HashMap::new())),
            workers: HashMap::new(),
            model_file,
        };

        m.start_new_workers(NUM_INIT_WORKERS).await?;
        Ok(m)
    }

    /// Start a new worker process on the local machine and connect to it
    #[tracing::instrument]
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

        let model_file = self.model_file.clone();

        // Start a new thread to spawn a new process
        let (pid, ch) = tokio::task::spawn(async move {
            // Start a new local worker process
            let command = format!("{} {}", port, model_file);
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

            /*
            let now = time::Instant::now();
            loop {
                if std::fs::metadata(format!("./tmp/{port}_ready")).is_ok() {
                    return Ok(pid);
                } else {
                    if now.elapsed().as_millis() >= WORKER_TIMEOUT {
                        return Err(anyhow!("timeout connecting to new worker process"));
                    }
                }
            }
            */

            /*
            // Connect to the new local worker
            // Spin until connection succeeds, or times out
            let now = time::Instant::now();
            loop {
                match WorkerClient::connect(format!("http://[::1]:{port}")).await {
                    Ok(conn) => {
                        return Ok((pid, conn));
                    }
                    Err(_) => {
                        if now.elapsed().as_millis() >= WORKER_TIMEOUT {
                            return Err(anyhow!("timeout connecting to new worker process"));
                        }
                        //tokio::time::sleep(std::time::Duration::from_millis(100)).await;
                        // sleep for 100 ms
                    }
                }
            }
            */

            let addr = format!("http://[::1]:{port}").parse::<Uri>().unwrap();
            //let ch = Endpoint::new(addr).unwrap().connect().await.unwrap();
            let endpoint = Endpoint::new(addr).unwrap();

            // Spin
            let now = time::Instant::now();
            loop {
                match endpoint.connect().await {
                    Ok(channel) => {
                        return Ok((pid, channel));
                    }
                    Err(_) => {
                        if now.elapsed().as_millis() >= WORKER_TIMEOUT {
                            return Err(anyhow!("timeout connecting to new worker process"));
                        }
                        //tokio::time::sleep(std::time::Duration::from_millis(100)).await;
                        // sleep for 100 ms
                    }
                }
            }
        })
        .await
        .unwrap()
        .unwrap();

        /*
        tokio::task::spawn(async move {
            // Spin until connection succeeds, or times out
            let now = time::Instant::now();
            loop {
                // Connect to the new local worker
                match WorkerClient::connect(format!("http://[::1]:{port}")).await {
                    Ok(conn) => return Ok(()),
                    Err(_) => {
                        if now.elapsed().as_millis() >= WORKER_TIMEOUT {
                            return Err(anyhow!("timeout connecting to new worker process"));
                        }
                        //tokio::time::sleep(std::time::Duration::from_millis(100)).await;
                        // sleep for 100 ms
                    }
                }
            }
        })
        .await
        .unwrap()
        .unwrap();
        */

        /*
        use tokio::time::{timeout, Duration};

        // Try to connect to the new local worker with a timeout
        let connect_future = WorkerClient::connect(format!("http://[::1]:{port}"));
        let timeout_duration = Duration::from_millis(WORKER_TIMEOUT as u64);
        match timeout(timeout_duration, connect_future).await {
            Ok(Ok(_)) => {
                // Connection succeeded
            }
            Ok(Err(e)) => {
                // Connection failed
                return Err(anyhow!("failed to connect to new worker process: {}", e));
            }
            Err(_) => {
                // Timeout
                return Err(anyhow!("timeout connecting to new worker process"));
            }
        }
        */

        let handle = Handle {
            port,
            pid,
            //conn,
            channel: ch,
        };

        info!("manager successfully connected to new worker (port = {port}, pid = {pid})",);

        self.workers
            .insert(handle.pid, (handle.clone(), WorkerStatus::Idle));
        Ok(handle)
    }

    /// Get a handle to an idle worker, or start a new worker if no workers are currently idle
    #[tracing::instrument]
    async fn get_idle_worker(&mut self) -> Result<Handle> {
        // Loop through all registered workers and return the first idle one
        match self
            .workers
            .values()
            .filter(|&(_, s)| s.clone() == WorkerStatus::Idle)
            .map(|(handle, _)| handle.clone())
            .collect::<Vec<Handle>>()
            .first()
        {
            Some(worker) => Ok(worker.clone()),
            None => {
                // If there are no idle workers, make a new worker (guaranteed to be idle)
                info!("all workers are busy: come back later");
                //self.start_new_worker().await
                Err(anyhow!(
                    "all workers are busy. chose not to make a new worker"
                ))
            }
        }
    }

    // ----- Interface ----- //

    /// Run inference on an idle worker
    #[tracing::instrument]
    pub async fn run_inference(&mut self, input: InputData) -> Result<Inference> {
        // Find an idle worker
        let handle = self.get_idle_worker().await?;
        debug!("make unlazy inference handle: {handle:?}");
        // Send req
        let req = Request::new(input.into());
        error!("built req");

        // Reconnent to the RPC client
        /*
        let mut conn = tokio::task::spawn(async move {
            let conn = WorkerClient::connect(format!("http://[::1]:{}", handle.port))
                .await
                .unwrap();
            error!("made conn");
            conn
        })
        .await
        .unwrap();
        */

        //error!("got conn");

        // Mark worker as busy
        self.workers
            .insert(handle.pid, (handle.clone(), WorkerStatus::Working));
        error!("marked as busy");

        // Send an RPC request for inference
        let mut conn = WorkerClient::new(handle.channel.clone());
        error!("made new worker client");
        let res = conn.image_inference(req).await?.into_inner().into();
        error!("sent infr req");
        // Ok(Inference::Text("some inference".to_string()))
        if !SPOT_WORKERS {
            self.workers
                .insert(handle.pid, (handle.clone(), WorkerStatus::Idle));
            error!("marked as unbusy");
        }
        Ok(res)
    }

    /// Get the statuses of all workers
    #[tracing::instrument]
    pub async fn all_status(&self) -> Result<HashMap<Handle, WorkerStatus>> {
        Ok(self.workers.iter().map(|(_, v)| v.clone()).collect())
    }

    // idea: the Manager is being cloned in the data part of the actix framework

    /// Return all the workers, without their status
    #[tracing::instrument]
    pub fn all_workers(&self) -> Vec<Handle> {
        self.workers
            .values()
            .map(|(w, _)| Handle {
                pid: w.pid,
                port: w.port,
                channel: w.channel.clone(),
            })
            .collect()
    }

    /// Start a new worker process on the local machine and connect to it
    #[tracing::instrument]
    pub async fn start_new_workers(&mut self, n: u16) -> Result<()> {
        let mut stream = tokio_stream::iter(0..n);
        while let Some(_) = stream.next().await {
            self.start_new_worker().await?;
        }
        Ok(())
    }

    /// Kill an idle worker and return a partial handle to it
    #[tracing::instrument]
    pub async fn kill_worker(&mut self) -> Result<Handle> {
        // Find an idle worker
        let handle = self.get_idle_worker().await?;
        let pid = handle.pid;
        // Remove it from the worker store
        let (handle, _) = self.workers.remove(&pid).unwrap();
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
