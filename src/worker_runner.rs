//! Entrypoint to start a worker locally

use autodep::worker::Worker;
use std::{env, process};

const USAGE: &str = "usage: ./worker <port> <torchscript file> ";

#[tokio::main]
async fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() - 1 != 2 {
        println!("{USAGE}");
        process::exit(1);
    }

    let port: u16 = args[1].parse().expect("invalid port");
    let model = args[2].clone();
    let worker = Worker::new(model, port).unwrap();
    worker.start().await.unwrap();
}
