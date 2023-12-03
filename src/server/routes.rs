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

async fn connect_to_worker(channel: Channel, input: InputData) -> Result<Inference> {
    let mut worker_client = WorkerClient::new(channel);
    let req = RpcRequest::new(input.into());
    let output: Inference = worker_client
        .image_inference(req)
        .await
        .unwrap()
        .into_inner()
        .into();

    Ok(output)
}

#[post("/image_inference")]
pub async fn image_inference(
    req: web::Json<protocol::B64Image>,
    state: web::Data<RwLock<Manager>>,
) -> Result<impl Responder> {
    let input = {
        InputData::Image(Image {
            image: general_purpose::STANDARD.decode(req.image.clone())?,
            height: None,
            width: None,
        })
    };

    let worker = {
        let mut manager = state.write().unwrap();
        let worker = manager
            .workers
            .values()
            .filter(|&(_, s)| s.clone() == WorkerStatus::Idle)
            .map(|(handle, _)| handle.clone())
            .collect::<Vec<Handle>>()
            .first()
            .cloned();

        match worker {
            Some(worker) => {
                manager.mark_working(worker.pid);
                Ok(worker)
            }
            None => {
                info!("all workers are busy: come back later");
                Err(anyhow!(
                    "all workers are busy. chose not to make a new worker"
                ))
            }
        }
    }?;

    let channel = worker.channel.clone();
    let output = connect_to_worker(channel, input).await?;

    {
        let mut manager = state.write().unwrap();
        manager.mark_idle(worker.pid);
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
    //let manager = state.read().unwrap();

    /*
    let status = manager.all_status().await;
    match status {
        Ok(s) => Ok(web::Json(protocol::AllStatusResponse(s))),
        Err(e) => Err(WebError { err: e }),
    }
    */

    let idling = {
        let manager = state.read().unwrap();
        (*manager)
            .workers
            .values()
            //.filter(|&(_, s)| s.clone() == WorkerStatus::Idle)
            .map(|(handle, status)| (handle.clone(), status.clone()))
            .collect::<Vec<(Handle, WorkerStatus)>>()
    };

    Ok(web::Json(idling))
}

/// HTTP request to get server statistics
#[get("/workers")]
pub async fn workers(_req: HttpRequest, state: web::Data<RwLock<Manager>>) -> impl Responder {
    let manager = state.read().unwrap();

    let workers = manager.all_workers();
    warn!("Workers: {workers:#?}");
    web::Json(workers)
}
