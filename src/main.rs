use actix_web::{middleware, web, App, HttpServer};
use autodep::manager::Manager;
use autodep::server::{self, routes};
use std::sync::{Arc, Mutex};
use std::{env, io, process};

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
    env::set_var(
        "RUST_LOG",
        "h2=warn,worker=debug,autodep=debug,actix_web=debug,actix_server=info",
    );
    //env::set_var("RUST_LOG", "debug");
    env_logger::init();

    let (model, port) = get_args();

    let manager = web::Data::new(Mutex::new(Manager::new(model.clone())));

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
