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
pub mod config {}

/// Network utility functions
pub mod util {
    use std::time::{SystemTime, UNIX_EPOCH};

    pub fn init_logging(log: &str) {
        std::env::set_var("RUST_LOG", log);
        tracing_subscriber::fmt::init();
    }

    pub fn init_libtorch(libtorch_path: &str) {
        let ld_lib_path = std::env::var("LD_LIBRARY_PATH").unwrap();
        std::env::set_var(
            "LD_LIBRARY_PATH",
            format!("${}:{}", ld_lib_path, libtorch_path),
        );
    }

    pub fn get_available_port() -> Option<u16> {
        port_scanner::request_open_port()
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

        pub fn load_image_from_disk(path: &str) -> crate::torch::InputData {
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
