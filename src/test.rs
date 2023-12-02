use anyhow::Result;
use autodep::manager;
use autodep::util::init_libtorch;
use autodep::util::test;
use env_logger;
use log::{debug, info, warn};
use std::{env, process};

const USAGE: &str = "usage: ./autodep <model file>";

use std::io::{stdin, stdout, Read, Write};

fn pause() {
    let mut stdout = stdout();
    stdout.write(b"Press Enter to continue...").unwrap();
    stdout.flush().unwrap();
    stdin().read(&mut [0]).unwrap();
}

/// This is test code right now
#[tokio::main]
async fn main() -> Result<()> {
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
    // TODO

    // Start the manager
    let mut manager = manager::Manager::new(model);

    // Allocate two workers
    manager.start_new_workers(2).await?;

    pause();

    //let killed = manager.kill_worker().await?;
    //debug!("KILLED {killed:#?}");

    // Get the statuses of all workers
    debug!("calling status");
    let status = manager.all_status().await?;
    info!("worker statuses: {:#?}", status);

    let img = test::get_test_image();
    let output = manager.run_inference(img).await?;
    info!("got output: {:#?}", output);

    Ok(())
}
