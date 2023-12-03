use crate::manager::Manager;
use actix_web::http::header::ContentType;
use actix_web::http::StatusCode;
use actix_web::HttpResponse;
use anyhow::anyhow;
use std::collections::HashMap;
use std::sync::Mutex;

mod protocol;
pub mod routes;

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
