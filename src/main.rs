use autodep::manager;
use autodep::util::init_libtorch;
use env_logger;
use log::{debug, info, warn};
use std::{env, process};

const USAGE: &str = "usage: ./autodep <model file>";

/// This is test code right now
#[tokio::main]
async fn main() {
    init_libtorch();
    env_logger::init();

    // Parse arguments
    let args: Vec<String> = env::args().collect();
    if args.len() - 1 != 1 {
        println!("{USAGE}");
        process::exit(1);
    }

    let model = args[1].clone();

    // Start the HTTP server

    // Start the manager
    let mut manager = manager::Manager::new(model);

    // Allocate two workers
    manager.start_new_worker().await.unwrap();
    manager.start_new_worker().await.unwrap();

    // Get the statuses of all workers
    debug!("calling status");
    let status = manager.all_status().await.unwrap();
    info!("worker statuses: {:#?}", status);
}
