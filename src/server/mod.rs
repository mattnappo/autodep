use crate::manager::Manager;
use actix_web::http::header::ContentType;
use actix_web::http::StatusCode;
use actix_web::HttpResponse;
use actix_web::{middleware, web, App, HttpServer};
use anyhow::anyhow;
use std::collections::HashMap;
use std::io;
use std::sync::RwLock;

mod protocol;
pub mod routes;

pub struct Server;

impl Server {
    pub async fn new(model: String, port: u16) -> io::Result<()> {
        let manager = web::Data::new(RwLock::new(Manager::new(model.clone()).await.unwrap()));

        // Start the HTTP server
        HttpServer::new(move || {
            App::new()
                .app_data(manager.clone())
                .wrap(middleware::Logger::default())
                .service(routes::inference)
                .service(routes::worker_status)
                .service(routes::all_workers)
                .service(routes::worker_info)
        })
        .bind(format!("0.0.0.0:{port}"))?
        .run()
        .await
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

impl From<base64::DecodeError> for WebError {
    fn from(err: base64::DecodeError) -> Self {
        WebError { err: anyhow!(err) }
    }
}
