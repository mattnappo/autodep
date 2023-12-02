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
use tonic::{Request, Response};

/// A handle to a worker
#[derive(Clone, Serialize)]
pub struct Handle {
    pub port: u16,
    pub pid: u32,
    #[serde(skip_serializing)]
    pub conn: Option<WorkerClient<Channel>>,
}

//unsafe impl Send for Handle {}
//unsafe impl Sync for Handle {}

/// The worker manager. Right now, assumes that all workers
/// are on the same host
pub struct Manager {
    /// Map from PID to `Handle`s of current workers
    //workers: Arc<RwLock<HashMap<u32, Handle>>>,
    workers: Arc<Mutex<HashMap<u32, Handle>>>,

    /// The path to the TorchScript model file
    model_file: String,
}

impl Manager {
    pub fn new(model_file: String) -> Self {
        error!("\n\nMADE A NEW MANAGER\n\n");
        Manager {
            //workers: Arc::new(RwLock::new(HashMap::new())),
            workers: Arc::new(Mutex::new(HashMap::new())),
            model_file,
        }
    }

    /// Start a new worker process on the local machine and connect to it
    async fn start_new_worker(&self) -> Result<Handle> {
        let mut workers = self.workers.lock().unwrap();
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

        let t = util::time();
        let out_name = format!("./logs/worker_{}_{}.out", port, t);
        let err_name = format!("./logs/worker_{}_{}.err", port, t);
        let out_log = File::create(out_name).expect("failed to open log");
        let err_log = File::create(err_name).expect("failed to open log");

        let pid = Command::new(WORKER_BINARY)
            .env("RUST_LOG", "debug,worker=debug,autodep=debug")
            .args(args)
            .stdout(out_log)
            .stderr(err_log)
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
        let x = workers.insert(handle.pid, handle.clone());
        warn!("\tupon add: {workers:?}");
        warn!("\tx is : {x:?}");
        Ok(handle)
    }

    /// Find an idle worker
    async fn find_idle_worker(&self) -> Result<Option<Handle>> {
        let workers = self.workers.lock().unwrap();
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
    pub async fn run_inference(&self, input: InputData) -> Result<Inference> {
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
            let handle = self.start_new_worker().await?;
            warn!("busy so created new handle: {handle:#?}");

            match handle.conn {
                Some(mut conn) => {
                    let req = Request::new(input.into());
                    let res = conn.image_inference(req).await?.into_inner().into();
                    Ok(res)
                }
                None => todo!(), // Try to connect again
            }
        }
    }

    /// Get the statuses of all workers
    pub async fn all_status(&self) -> Result<HashMap<Handle, WorkerStatus>> {
        let mut map: HashMap<Handle, WorkerStatus> = HashMap::new();

        let workers = self.workers.lock().unwrap();
        //let mut handles = tokio_stream::iter(workers.values().collect::<Vec<&Handle>>());
        let mut handles = tokio_stream::iter(workers.values());
        while let Some(handle) = handles.next().await {
            debug!("getting status of worker pid {}", handle.pid);
            let conn = handle.conn.clone();
            let req = Request::new(rpc::Empty {});
            let res = conn.unwrap().get_status(req).await?.into_inner();
            //map.insert(handle.clone(), res.into());
            debug!("got single status res: {res:?}");
            map.insert(handle.clone(), res.into());
        }

        error!("THE MAP IS: {map:?}");

        Ok(map)
    }

    // idea: the Manager is being cloned in the data part of the actix framework

    /// Return all the workers, without their status
    pub fn all_workers(&self) -> Vec<Handle> {
        let workers = self.workers.lock().unwrap();
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
    pub async fn kill_worker(&self) -> Result<Handle> {
        // Find an idle worker
        match self.find_idle_worker().await? {
            // If that worker exists, kill it
            Some(h) => {
                let pid = h.pid;
                // Remove it from the worker store
                let mut workers = self.workers.lock().unwrap();
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
