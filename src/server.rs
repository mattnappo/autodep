//! The user-facing JSON web server that listens for inference requests. This
//! is the "front end". The inference route is automatically created, and
//! distributes inference computation across the array of workers.

use crate::config::*;
use crate::manager::Manager;
use actix_web::http::header::ContentType;
use actix_web::http::StatusCode;
use actix_web::{get, web, HttpRequest, HttpResponse, Responder};
use log::debug;
use protocol::*;
use std::sync::Mutex;

type Result<T> = std::result::Result<T, WebError>;

// tmp
use crate::manager::Handle;
use crate::worker::WorkerStatus;
use std::collections::HashMap;

mod protocol {
    use crate::manager::Handle;
    use crate::worker::WorkerStatus;
    use serde::ser::{Serialize, SerializeMap, Serializer};
    use std::collections::HashMap;

    pub struct AllStatusResponse(pub HashMap<Handle, WorkerStatus>);

    impl Serialize for AllStatusResponse {
        fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
        where
            S: serde::Serializer,
        {
            let mut map = serializer.serialize_map(Some(self.0.len()))?;
            for (k, v) in &self.0 {
                map.serialize_entry(&format!("{k:?}"), &format!("{v:?}"))?;
            }
            map.end()
        }
    }

    impl From<HashMap<Handle, WorkerStatus>> for AllStatusResponse {
        fn from(map: HashMap<Handle, WorkerStatus>) -> Self {
            Self(map)
        }
    }
}

pub struct Server {
    manager: Mutex<Manager>,
}

impl Server {
    pub fn new(model_file: String) -> Result<Self> {
        Ok(Server {
            manager: Mutex::new(Manager::new(model_file)),
        })
    }
}

/// Handle HTTP request for inference
#[get("/inference")]
pub async fn inference(_req: HttpRequest, state: web::Data<Server>) -> Result<impl Responder> {
    //let manager = state.manager.lock().unwrap();

    Ok(web::Json("ok"))
}

/// HTTP request to get server statistics
#[get("/manager_info")]
pub async fn manager_info(_req: HttpRequest, state: web::Data<Server>) -> Result<impl Responder> {
    let mut manager = state.manager.lock().unwrap();

    let status = manager.all_status().await;

    match status {
        Ok(s) => Ok(web::Json(format!("{s:?}"))),
        Err(e) => Err(WebError { err: e }),
        //Err(e) => Ok(web::Json("some error lol".to_string())),
    }
}

#[derive(Debug)]
pub struct WebError {
    err: anyhow::Error,
}

impl std::fmt::Display for WebError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.err)
    }
}

impl actix_web::error::ResponseError for WebError {
    fn error_response(&self) -> HttpResponse {
        println!("weird don't run plz");
        let err = HashMap::from([("errors", vec![self.to_string()])]);

        HttpResponse::build(self.status_code())
            .insert_header(ContentType::json())
            .json(err)
    }

    fn status_code(&self) -> StatusCode {
        StatusCode::INTERNAL_SERVER_ERROR
    }
}

impl From<anyhow::Error> for WebError {
    fn from(err: anyhow::Error) -> WebError {
        WebError { err }
    }
}
