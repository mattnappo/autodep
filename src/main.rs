use autodep::manager;
use autodep::util::init_libtorch;
use std::{env, process};

const USAGE: &str = "usage: ./autodep <model file>";

/// This is test code right now
#[tokio::main]
async fn main() {
    init_libtorch();

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
    let handle1 = manager.start_new_worker().await.unwrap();
    dbg!(handle1);

    std::thread::sleep(std::time::Duration::from_millis(5000));

    let handle2 = manager.start_new_worker().await.unwrap();
    dbg!(handle2);

    // Get the statuses of all workers
    let status = manager.all_status().await.unwrap();
    dbg!(status);
}
