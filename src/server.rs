//! The user-facing JSON web server that listens for inference requests. This
//! is the "front end". The inference route is automatically created, and
//! distributes inference computation across the array of workers.

use crate::config::*;
use crate::manager::Manager;
use actix_web::http::header::ContentType;
use actix_web::{get, web, HttpRequest, HttpResponse, Responder, Result};
use log::debug;
use protocol::*;
use std::sync::Mutex;

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

#[derive(Debug)]
pub struct Server {
    manager: Manager,
}

pub struct Empty {
    s: String,
}

impl Empty {
    pub fn new(s: String) -> Self {
        Self { s }
    }
}

impl Server {
    pub fn new(model_file: String) -> Result<Self> {
        Ok(Server {
            manager: Manager::new(model_file),
        })
    }
}

/// Handle HTTP request for inference
#[get("/inference")]
pub async fn inference(_req: HttpRequest, state: web::Data<Server>) -> Result<impl Responder> {
    //pub async fn inference(_req: HttpRequest, state: web::Data<Server>) -> Result<impl Responder> {
    //let manager = state.manager.lock().unwrap();
    //manager.run_inference();

    debug!("got request {:#?}", _req);

    debug!("state: {state:#?}");

    let test_handle = Handle {
        port: 123,
        pid: 456,
        conn: None,
    };

    let test = (test_handle, WorkerStatus::Working);
    let map = HashMap::from([test]);

    /*
    HttpResponse::Ok()
        .content_type(ContentType::json())
        //.json(AllStatusResponse(map))
        .json(vec![1, 2, 3])
    */

    Ok(web::Json(AllStatusResponse(map)))
}

/// HTTP request to get server statistics
#[get("/manager_info")]
pub async fn manager_info(_req: HttpRequest, state: web::Data<Server>) -> HttpResponse {
    let mut manager = state.manager.lock().unwrap();

    match manager.all_status().await {
        Ok(status) => HttpResponse::Ok()
            .content_type(JSON)
            .json(Into::<AllStatusResponse>::into(status)),
        Err(e) => HttpResponse::InternalServerError()
            .content_type(JSON)
            .json(format!("{e:?}")),
    }
}
