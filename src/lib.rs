pub mod manager;
pub mod torch;
pub mod worker;

/// The worker's RPC server
pub mod rpc {
    tonic::include_proto!("worker");
}

/// Autodep configuration -- can eventually be lazy_static parsed from a config
/// file
pub mod config {
    /// Path to the local libtorch installation
    pub const LIBTORCH_PATH: &str = "/home/matt/rust/autodep/target/debug/build/torch-sys-ff2ab40729eb7ad5/out/libtorch/libtorch/lib";

    /// The path to the compiled worker binary
    pub const WORKER_BINARY: &str = "./target/debug/worker";

    /// Maximum number of workers
    pub const MAX_WORKERS: usize = 20;

    /// Pick `TOP_N` largest softmax probabilities in a classifier model
    pub const TOP_N: i64 = 5;
}

/*
#[derive(Debug)]
struct GenericError(&'static str);

impl std::fmt::Display for GenericError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Error: {}", self.0)
    }
}

impl std::error::Error for GenericError {}
*/

/// Network utility functions
pub mod util {
    pub fn get_available_ports(n: u16) -> Option<Vec<u16>> {
        let ports = (8000..(12000))
            .filter(|port| port_is_available(*port))
            .take(n as usize)
            .collect::<Vec<u16>>();

        match ports.len() == 4 {
            true => Some(ports),
            _ => None,
        }
    }

    /*
    pub fn get_available_port() -> Option<u16> {
        (8000..(12000))
            .filter(|port| port_is_available(*port))
            .collect::<Vec<u16>>()
            .first()
            .cloned()
    }
    */

    pub fn get_available_port() -> Option<u16> {
        port_scanner::request_open_port()
    }

    fn port_is_available(port: u16) -> bool {
        match std::net::TcpListener::bind(("127.0.0.1", port)) {
            Ok(_) => true,
            Err(_) => false,
        }
    }

    pub fn init_libtorch() {
        let ld_lib_path = std::env::var("LD_LIBRARY_PATH").unwrap();
        std::env::set_var(
            "LD_LIBRARY_PATH",
            format!("${}:{}", ld_lib_path, super::config::LIBTORCH_PATH),
        );
        std::env::set_var("RUST_LOG", "debug");
    }
}
