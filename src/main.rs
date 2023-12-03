use actix_web::{middleware, web, App, HttpServer};
use autodep::config::RUST_LOG;
use autodep::manager::Manager;
use autodep::server::{self, routes};
use autodep::util::init_libtorch;
use std::sync::RwLock;
use std::sync::{Arc, Mutex};
use std::{env, io, process};

use tracing::info;
use tracing_subscriber;

const USAGE: &str = "usage: ./autodep <port> <model file> ";

fn get_args() -> (String, u16) {
    let args: Vec<String> = env::args().collect();
    if args.len() - 1 != 2 {
        println!("{USAGE}");
        process::exit(1);
    }

    let port: u16 = args[1].parse().expect("invalid port");
    let model = args[2].clone();

    (model, port)
}

#[actix_web::main]
async fn main() -> io::Result<()> {
    //init_libtorch();
    env::set_var("RUST_LOG", RUST_LOG);
    for (k, v) in env::vars() {
        println!("export {k}='{v}'");
    }
    //env_logger::init();
    tracing_subscriber::fmt::init();

    let (model, port) = get_args();

    let manager = web::Data::new(RwLock::new(Manager::new(model.clone()).await.unwrap()));

    // Start the HTTP server
    HttpServer::new(move || {
        App::new()
            .app_data(manager.clone())
            .wrap(middleware::Logger::default())
            .service(routes::image_inference)
            .service(routes::workers)
            .service(routes::worker_status)
    })
    .bind(format!("0.0.0.0:{port}"))?
    .run()
    .await
}
