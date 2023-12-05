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
    //pub const LIBTORCH_PATH: &str = "/home/matt/rust/autodep/target/debug/build/torch-sys-3f99fa23d8dcb15b/out/libtorch/libtorch/lib";
    pub const LIBTORCH_PATH: &str = "torch-sys-abdb1e401c3e2cb9/out/libtorch/libtorch/lib/";

    /// Logging settings
    pub const RUST_LOG: &str =
        "h2=debug,worker=debug,autodep=debug,actix_web=debug,actix_server=debug";

    /// The path to the compiled worker binary
    //pub const WORKER_BINARY: &str = "./target/debug/worker";
    pub const WORKER_BINARY: &str = "./target/release/worker";

    /// Maximum number of workers
    pub const MAX_WORKERS: usize = 20;

    /// Number of workers to start the server with
    pub const NUM_INIT_WORKERS: u16 = 15;

    /// Max time given to connect to a worker's RPC server, in millis
    pub const WORKER_TIMEOUT: u128 = 2000;

    /// Spot workers are one-time-use workers
    pub const SPOT_WORKERS: bool = false;

    /// Dynamically allocate new worker processes when necessary
    pub const AUTO_SCALE: bool = false;

    /// When FAST_WORKERS is true, workers do not get set as `Working`.
    /// Instead, they always appear as `Idle`.
    pub const FAST_WORKERS: bool = true;
}

/// Network utility functions
pub mod util {
    use std::time::{SystemTime, UNIX_EPOCH};

    pub fn init_logging() {
        std::env::set_var("RUST_LOG", super::config::RUST_LOG);
        tracing_subscriber::fmt::init();
    }

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

            crate::torch::InputData::B64Image(
                crate::torch::Image {
                    image,
                    height: None,
                    width: None,
                }
                .into(),
            )
        }

        pub fn load_image_from_disk(path: String) -> crate::torch::InputData {
            let mut file = std::fs::File::open(path).unwrap();
            let mut image: Vec<u8> = vec![];
            file.read_to_end(&mut image).unwrap();

            crate::torch::InputData::B64Image(
                crate::torch::Image {
                    image,
                    height: None,
                    width: None,
                }
                .into(),
            )
        }
    }
}
