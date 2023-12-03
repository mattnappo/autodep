//! Entrypoint to start a worker locally

use autodep::config::*;
use autodep::util::init_libtorch;
use autodep::worker::Worker;
use std::{env, process};
use tokio::fs;

const USAGE: &str = "usage: ./worker <port> <model file> ";

#[tokio::main]
async fn main() {
    init_libtorch();
    std::env::set_var("RUST_LOG", RUST_LOG);
    //env_logger::init();
    tracing_subscriber::fmt::init();

    let args: Vec<String> = env::args().collect();
    if args.len() - 1 != 2 {
        println!("{USAGE}");
        process::exit(1);
    }

    let port: u16 = args[1].parse().expect("invalid port");
    let model = args[2].clone();
    let worker = Worker::new(model, port).unwrap();

    let serve = tokio::spawn(async move {
        worker.start().await.unwrap();
    });

    tokio::join!(serve, fs::write(format!("./tmp/{port}_ready"), ""));
}
