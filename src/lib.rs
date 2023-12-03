pub mod manager;
pub mod server;
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
    //"/home/matt/autodep/target/debug/build/torch-sys-ff332d9a0497eb5d/out/libtorch/libtorch/lib/";

    /// The path to the compiled worker binary
    pub const WORKER_BINARY: &str = "./target/debug/worker";

    /// Maximum number of workers
    pub const MAX_WORKERS: usize = 20;

    /// in ms
    pub const WORKER_TIMEOUT: u128 = 2000;

    pub const RUST_LOG: &str =
        "h2=debug,worker=debug,autodep=debug,actix_web=debug,actix_server=debug";

    /// Number of workers to start the server with
    pub const NUM_INIT_WORKERS: u16 = 2;

    /// Pick `TOP_N` largest softmax probabilities in a classifier model
    pub const TOP_N: i64 = 5;
}

/// Network utility functions
pub mod util {
    use std::time::{SystemTime, UNIX_EPOCH};

    pub fn get_available_port() -> Option<u16> {
        port_scanner::request_open_port()
    }

    pub fn init_libtorch() {
        let ld_lib_path = std::env::var("LD_LIBRARY_PATH").unwrap();
        std::env::set_var(
            "LD_LIBRARY_PATH",
            format!("${}:{}", ld_lib_path, super::config::LIBTORCH_PATH),
        );
    }

    pub fn time() -> u64 {
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs()
    }

    /// Functions for testing purposes
    pub mod test {
        use std::io::Read;
        /// Get a test image
        pub fn get_test_image() -> crate::torch::InputData {
            let mut file = std::fs::File::open("images/cat.png").unwrap();
            let mut image: Vec<u8> = vec![];
            file.read_to_end(&mut image).unwrap();

            crate::torch::InputData::Image(crate::torch::Image {
                image,
                height: None,
                width: None,
            })
        }
    }
}
