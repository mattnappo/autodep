//! The user-facing JSON web server that listens for inference requests. This
//! is the "front end". The inference route is automatically created, and
//! distributes inference computation across the array of workers.

use super::protocol;
use super::WebError;
use crate::config::*;
use crate::manager::Handle;
use crate::manager::Manager;
use crate::rpc::worker_client::WorkerClient;
use crate::torch::Inference;
use crate::torch::{Image, InputData};
use crate::worker::WorkerStatus;
use actix_web::http::header::ContentType;
use actix_web::http::StatusCode;
use actix_web::{get, post, web, HttpRequest, HttpResponse, Responder};
use anyhow::anyhow;
use base64::{
    alphabet,
    engine::{self, general_purpose},
    Engine as _,
};
use log::*;
use std::sync::Mutex;
use std::sync::RwLock;
use tonic::{transport::Channel, Request as RpcRequest, Response as RpcResponse};

type Result<T> = std::result::Result<T, WebError>;

#[post("/image_inference")]
pub async fn image_inference(
    req: web::Json<protocol::B64Image>,
    state: web::Data<RwLock<Manager>>,
) -> Result<impl Responder> {
    // Parse the input request
    let input = {
        InputData::Image(Image {
            image: general_purpose::STANDARD.decode(req.image.clone())?,
            height: None,
            width: None,
        })
    };

    // Get a handle to an idle worker
    let worker = {
        let mut manager = state.write().unwrap();
        let worker = manager.get_idle_worker();

        match worker {
            Some(worker) => {
                manager.set_worker_status(worker.pid, WorkerStatus::Working);
                Ok(worker)
            }
            None => {
                warn!("all workers are busy: retry again later");
                Err(anyhow!("all workers are busy: retry again later"))
            }
        }
    }?;

    // Send the inference request to the worker via RPC
    let channel = worker.channel.clone();
    let output = Manager::run_inference(channel, input).await?;

    // Mark the worker as Idle again
    {
        let mut manager = state.write().unwrap();
        manager.set_worker_status(worker.pid, WorkerStatus::Idle);
    }

    info!("finished serving inference request");

    Ok(web::Json(output))
}

/// HTTP request to get the status of all workers
#[get("/workers/status")]
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

/// HTTP request to get server statistics
#[get("/workers")]
pub async fn workers(_req: HttpRequest, state: web::Data<RwLock<Manager>>) -> impl Responder {
    let manager = state.read().unwrap();

    let workers = manager.workers();
    web::Json(workers)
}
