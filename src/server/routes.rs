//! The user-facing JSON web server that listens for inference requests. This
//! is the "front end". The inference route is automatically created, and
//! distributes inference computation across the array of workers.

use super::WebError;

use crate::manager::{self, Manager};

use crate::torch::{Image, InputData};
use crate::worker::WorkerStatus;
use crate::{config, torch};

use actix_web::{get, post, web, HttpRequest, Responder};
use anyhow::anyhow;
use base64::{engine::general_purpose, Engine as _};
use tracing::*;

use std::sync::RwLock;

type Result<T> = std::result::Result<T, WebError>;

#[post("/inference")]
pub async fn inference(
    req: web::Json<torch::InferenceTask>,
    state: web::Data<RwLock<Manager>>,
) -> Result<impl Responder> {
    // Parse the input request
    let input = req.into_inner();
    info!("got inference request: {:?}", input);

    // Get a handle to an idle worker
    let worker = {
        let mut manager = state.write().unwrap();
        let worker = manager.get_idle_worker();

        debug!("found idle worker");

        match worker {
            Some(worker) => {
                manager.set_worker_status(worker.pid, WorkerStatus::Working);
                debug!("set idle worker to busy");
                Ok::<manager::Handle, anyhow::Error>(worker)
            }
            None => {
                warn!("all workers are busy");
                if config::AUTO_SCALE {
                    // Start a new worker
                    info!("dynamically starting a new worker");
                    manager.start_new_workers(1).await?;
                    let worker = {
                        let worker = manager.get_idle_worker();
                        if let Some(worker) = worker {
                            manager.set_worker_status(worker.pid, WorkerStatus::Working);
                            debug!("set idle worker to busy");
                            Ok(worker)
                        } else {
                            Err(anyhow!(
                                "all workers are busy, even after starting a new worker"
                            ))
                        }
                    }?;

                    Ok(worker)
                } else {
                    Err(anyhow!("all workers are busy: retry again later"))
                }
            }
        }
    }?;

    // Send the inference request to the worker via RPC
    let channel = worker.channel.clone();
    debug!("sending inference request");
    let output = Manager::run_inference(channel, input).await?;
    debug!("received inference response");

    // Mark the worker as Idle again
    {
        let mut manager = state.write().unwrap();
        manager.set_worker_status(worker.pid, WorkerStatus::Idle);
    }

    info!("finished serving inference request");

    Ok(web::Json(output))
}

/// HTTP request to get the status of all workers
#[get("/workers/_status")]
pub async fn worker_status(
    _req: HttpRequest,
    state: web::Data<RwLock<Manager>>,
) -> Result<impl Responder> {
    let status = {
        let manager = state.read().unwrap();
        manager.all_status().unwrap()
    };

    Ok(web::Json(status))
}

/// HTTP request to get all Working workers
#[get("/workers")]
pub async fn all_workers(_req: HttpRequest, state: web::Data<RwLock<Manager>>) -> impl Responder {
    let workers = {
        let manager = state.read().unwrap();
        manager.all_workers().unwrap()
    };

    web::Json(workers)
}

/// HTTP request to get server statistics
#[get("/workers/_info")]
pub async fn worker_info(_req: HttpRequest, state: web::Data<RwLock<Manager>>) -> impl Responder {
    let manager = state.read().unwrap();

    let workers = manager.workers();
    web::Json(workers)
}
