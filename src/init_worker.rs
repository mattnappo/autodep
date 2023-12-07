//! Entrypoint to start a worker locally

use autodep::util::init_libtorch;
use autodep::worker::Worker;
use config::{Config, File};
use std::{env, process};

const USAGE: &str = "usage: ./worker <port> <config file> <model file>";

fn get_args() -> (String, Config, u16) {
    let args: Vec<String> = env::args().collect();
    if args.len() - 1 != 3 {
        println!("{USAGE}");
        process::exit(1);
    }

    let port: u16 = args[1].parse().unwrap();
    let config_file = &args[2];
    let model_file = &args[3];

    let config = Config::builder()
        .add_source(File::with_name(&config_file))
        .build()
        .unwrap();

    (model_file.into(), config, port)
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let (model_file, config, port) = get_args();

    init_libtorch(&config.get_string("worker.libtorch_path").unwrap());
    std::env::set_var("RUST_LOG", config.get_string("manager.logging").unwrap());
    tracing_subscriber::fmt::init();

    let worker = Worker::new(&model_file, port).unwrap();

    worker.start().await
}
