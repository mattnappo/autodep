use autodep::server::Server;
use config::Config;
use config::File;
use std::{env, io, process};

use autodep::util;

const USAGE: &str = "usage: ./autodep <config file> <model file>";

fn get_args() -> (String, Config) {
    let args: Vec<String> = env::args().collect();
    if args.len() - 1 != 2 {
        println!("{USAGE}");
        process::exit(1);
    }

    let config_file = &args[1];
    let model_file = &args[2];

    let config = Config::builder()
        .add_source(File::with_name(&config_file))
        .build()
        .unwrap();

    (model_file.into(), config)
}

#[actix_web::main]
async fn main() -> io::Result<()> {
    let (model, config) = get_args();
    util::init_logging(&config.get_string("manager.logging").unwrap());

    Server::new(&model, config).await
}
