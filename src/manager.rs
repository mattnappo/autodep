//! The manager (worker manager, resource allocator, allocator) is responsible for
//! interfacing with a set of workers. The manager starts and stops workers, and
//! forwards inference requests

use crate::rpc;
use crate::rpc::worker_client::WorkerClient;
use crate::torch;
use crate::util;
use crate::worker::WorkerStatus;
use anyhow::anyhow;
use anyhow::Result;
use config::Config;
use rand::seq::SliceRandom;

use serde::ser::Serialize;
use serde::Serialize as DeriveSerialize;
use std::collections::HashMap;
use std::fs::File;
use std::process::Command;

use std::time;

use tokio_stream::StreamExt;
use tonic::transport::Channel;
use tonic::transport::Endpoint;
use tonic::transport::Uri;
use tonic::Request;
use tracing::*;

/// A handle to a worker
#[derive(Clone)]
pub struct Handle {
    pub port: u16,
    pub pid: u32,
    pub channel: Channel,
}

impl Handle {
    pub fn partial(&self) -> PartialHandle {
        PartialHandle {
            pid: self.pid,
            port: self.port,
        }
    }
}

/// A (pid, port) tuple
#[derive(Clone, Debug, DeriveSerialize, Eq, PartialEq, Hash)]
pub struct PartialHandle {
    pub port: u16,
    pub pid: u32,
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
    workers: HashMap<u32, (Handle, WorkerStatus)>,

    /// The path to the TorchScript model file
    model_file: String,

    /// System configuration
    pub config: Config,
}

impl Manager {
    /// Start a new manager and start `NUM_INIT_WORKERS` new worker processes
    pub async fn new(model_file: &str, config: Config) -> Result<Self> {
        let mut m = Manager {
            workers: HashMap::new(),
            model_file: model_file.into(),
            config: config.clone(),
        };

        m.start_new_workers(config.get_int("manager.num_init_workers")? as u16)
            .await?;
        Ok(m)
    }

    /// Start a new worker process on the local machine and connect to it
    //#[tracing::instrument]
    async fn start_new_worker(&mut self) -> Result<Handle> {
        if self.workers.len() + 1 >= self.config.get_int("manager.max_workers")? as usize {
            return Err(anyhow!(
                "maximum number of workers exceeded. cannot allocate any more",
            ));
        }

        // Find an open port
        let port = util::get_available_port().unwrap(); // Use ok_or here
        debug!("found free port {port}");

        let model_file = self.model_file.clone();

        // Start a new thread to spawn a new process
        let cfg = self.config.clone();
        let (pid, ch) = tokio::task::spawn(async move {
            // Forward worker's logs to a file
            let t = util::time();
            std::fs::create_dir_all("logs/")?;
            let out_name = format!("./logs/worker_{}_{}.out", port, t);
            let err_name = format!("./logs/worker_{}_{}.err", port, t);
            let out_log = File::create(out_name).expect("failed to open log");
            let err_log = File::create(err_name).expect("failed to open log");

            // Spawn the new worker process
            let config_file = std::env::args().collect::<Vec<String>>();
            let config_file = config_file.get(1).unwrap();
            let command = format!("{} {} {}", port, config_file, model_file);
            let args = command.split(' ').map(|n| n.to_string());
            let pid = Command::new(cfg.get_string("worker.binary")?)
                .env("RUST_LOG", cfg.get_string("manager.logging")?)
                .args(args)
                .stdout(out_log)
                .stderr(err_log)
                .spawn()
                .unwrap()
                .id();

            info!("manager started new worker process {pid}");

            // Build the rpc endpoint
            let endpoint =
                Endpoint::new(format!("http://[::1]:{port}").parse::<Uri>().unwrap()).unwrap();

            // Spin until the connection succeeds
            let now = time::Instant::now();
            loop {
                match endpoint.connect().await {
                    Ok(channel) => {
                        return Ok((pid, channel));
                    }
                    Err(_) => {
                        if now.elapsed().as_millis()
                            >= cfg.get_int("manager.worker_timeout")? as u128
                        {
                            return Err(anyhow!("timeout connecting to new worker process"));
                        }
                        tokio::time::sleep(std::time::Duration::from_millis(100)).await;
                    }
                }
            }
        })
        .await
        .unwrap()?;

        let handle = Handle {
            port,
            pid,
            channel: ch,
        };

        info!("manager successfully connected to new worker (port = {port}, pid = {pid})",);

        self.workers
            .insert(handle.pid, (handle.clone(), WorkerStatus::Idle));
        Ok(handle)
    }

    // ----- Interface ----- //

    /// Set the status of a worker
    pub fn set_worker_status(&mut self, pid: u32, status: WorkerStatus) {
        let (h, _) = self.workers.get(&pid).unwrap();
        self.workers.insert(pid, (h.clone(), status));
    }

    /// Run inference on a worker given an RPC channel to the worker
    pub async fn run_inference(
        channel: Channel,
        input: torch::InferenceTask,
    ) -> Result<torch::TimedInference> {
        let mut worker_client = WorkerClient::new(channel);
        let ty = input.inference_type.clone();
        let req = Request::new(input.into());
        let rpc_output: rpc::Inference = worker_client.compute_inference(req).await?.into_inner();

        // Parse output
        let output = match ty {
            torch::InferenceType::ImageClassification { .. } => {
                let classes: Vec<torch::Class> = rpc_output.classification.unwrap().into();
                torch::Inference::Classification(classes)
            }
            torch::InferenceType::ImageToImage => {
                torch::Inference::B64Image(rpc_output.image.unwrap().into())
            }
            _ => unimplemented!(),
        };

        Ok((output, time::Duration::from_secs_f32(rpc_output.duration)))
    }

    /// Get a handle to an idle worker, if any workers are idle
    pub fn get_idle_worker(&self) -> Option<Handle> {
        self.workers
            .values()
            .filter(|&(_, s)| s.clone() == WorkerStatus::Idle)
            .map(|(handle, _)| handle.clone())
            .collect::<Vec<Handle>>()
            .first()
            //.choose(&mut rand::thread_rng())
            .cloned()
    }

    /// Get the statuses of all workers
    // #[tracing::instrument]
    pub fn all_status(&self) -> Result<HashMap<Handle, WorkerStatus>> {
        Ok(self
            .workers
            .values()
            .map(|(handle, status)| (handle.clone(), status.clone()))
            .collect())
    }

    // #[tracing::instrument]
    pub fn all_workers(&self) -> Result<HashMap<Handle, WorkerStatus>> {
        Ok(self
            .all_status()
            .unwrap()
            .into_iter()
            .filter(|(_, s)| *s == WorkerStatus::Working)
            .collect())
    }

    /// Return all the workers, without their status
    // #[tracing::instrument]
    pub fn workers(&self) -> Vec<PartialHandle> {
        self.workers
            .values()
            .map(|(w, _)| PartialHandle {
                pid: w.pid,
                port: w.port,
            })
            .collect()
    }

    /// Get statistics of all workers
    // #[tracing::instrument]
    pub async fn all_stats(&self) -> Result<HashMap<PartialHandle, u64>> {
        let mut map: HashMap<PartialHandle, u64> = HashMap::new();

        let mut handles = tokio_stream::iter(self.workers.values());
        while let Some((handle, _)) = handles.next().await {
            debug!("getting status of worker pid {}", handle.pid);
            let channel = handle.channel.clone();

            let req = Request::new(rpc::Empty {});
            let mut worker_client = WorkerClient::new(channel);
            let res = worker_client.get_stats(req).await?.into_inner();

            map.insert(handle.partial(), res.reqs_served);
        }

        Ok(map)
    }

    /// Start a new worker process on the local machine and connect to it
    //#[tracing::instrument]
    pub async fn start_new_workers(&mut self, n: u16) -> Result<()> {
        let mut stream = tokio_stream::iter(0..n);
        while let Some(_) = stream.next().await {
            self.start_new_worker().await?;
        }
        Ok(())
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
        // Skip hashing the channel
        (&self.port, &self.pid).hash(state);
    }
}
impl PartialEq for Handle {
    fn eq(&self, other: &Self) -> bool {
        self.port == other.port && self.pid == other.pid
    }
}
impl Eq for Handle {}
