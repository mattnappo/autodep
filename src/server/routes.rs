//! The user-facing JSON web server that listens for inference requests. This
//! is the "front end". The inference route is automatically created, and
//! distributes inference computation across the array of workers.

use super::WebError;

use crate::manager::{self, Manager};

use crate::rpc;
use crate::rpc::worker_client::WorkerClient;
use tonic::Request;

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
        let worker = {
            let m = state.read().unwrap();
            m.get_idle_worker()
        };

        debug!("found idle worker");

        match worker {
            Some(worker) => Ok::<manager::Handle, anyhow::Error>(worker),
            None => {
                warn!("all workers are busy");
                Err(anyhow!("all workers are busy"))
                /*
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
                */
            }
        }
    }?;

    // Send the inference request to the worker via RPC
    let channel = worker.channel.clone();
    debug!("sending inference request");

    // Mark the work as busy
    let fast_workers = {
        let s = state.read().unwrap();
        s.config.get_bool("manager.fast_workers")?
    };

    //let output = Manager::run_inference(channel, input).await?;

    let mut worker_client = WorkerClient::new(channel);
    let ty = input.inference_type.clone();
    let req = Request::new(input.into());

    if !fast_workers {
        let mut manager = state.write().unwrap();
        manager.set_worker_status(worker.pid, WorkerStatus::Working);
        debug!("set idle worker to busy");
    }
    let rpc_output: rpc::Inference = worker_client
        .compute_inference(req)
        .await
        .unwrap()
        .into_inner();

    // Mark the worker as Idle again
    {
        let mut manager = state.write().unwrap();
        manager.set_worker_status(worker.pid, WorkerStatus::Idle);
    }

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

    let res = (
        output,
        std::time::Duration::from_secs_f32(rpc_output.duration),
    );

    debug!("received inference response");

    info!("finished serving inference request");

    //Ok(web::Json(output))
    Ok(web::Json(res))
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
pub async fn worker_info(
    _req: HttpRequest,
    state: web::Data<RwLock<Manager>>,
) -> Result<impl Responder> {
    let manager = state.read().unwrap();
    let stats = manager.all_stats().await?;
    let stats_list = stats.into_iter().collect::<Vec<_>>();
    Ok(web::Json(stats_list))
}
