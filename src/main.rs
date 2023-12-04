use autodep::server::Server;
use std::{env, io, process};

use autodep::util;

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
    util::init_logging();

    let (model, port) = get_args();
    Server::new(model, port).await
}
