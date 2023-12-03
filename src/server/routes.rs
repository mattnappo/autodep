//! The user-facing JSON web server that listens for inference requests. This
//! is the "front end". The inference route is automatically created, and
//! distributes inference computation across the array of workers.

use super::protocol;
use super::WebError;
use crate::config::*;
use crate::manager::Manager;
use crate::torch::{Image, InputData};
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

type Result<T> = std::result::Result<T, WebError>;

/// Handle HTTP request for inference
#[post("/image_inference")]
pub async fn image_inference(
    req: web::Json<protocol::B64Image>,
    state: web::Data<Mutex<Manager>>,
) -> Result<impl Responder> {
    // Get input from request
    let input = {
        InputData::Image(Image {
            image: general_purpose::STANDARD.decode(req.image.clone())?,
            height: None,
            width: None,
        })
    };

    // Tell the manager to compute inference
    let mut manager = state.lock().unwrap();
    let output = manager.run_inference(input).await?;
    //actix_web::rt::task::spawn_blocking(move || manager.run_inference(input).await).await;

    info!("finished serving inference request");

    Ok(web::Json(output))
}

/// HTTP request to get the status of all workers
#[get("/workers/status")]
pub async fn worker_status(
    _req: HttpRequest,
    state: web::Data<Mutex<Manager>>,
) -> Result<impl Responder> {
    let manager = state.lock().unwrap();

    let status = manager.all_status().await;

    match status {
        Ok(s) => Ok(web::Json(protocol::AllStatusResponse(s))),
        Err(e) => Err(WebError { err: e }),
    }
}

/// HTTP request to get server statistics
#[get("/workers")]
pub async fn workers(_req: HttpRequest, state: web::Data<Mutex<Manager>>) -> impl Responder {
    let manager = state.lock().unwrap();

    let workers = manager.all_workers();
    warn!("Workers: {workers:#?}");
    web::Json(workers)
}
