pub mod manager;
pub mod torch;
pub mod worker;

/// The worker's RPC server
pub mod rpc {
    tonic::include_proto!("worker");
}

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

    pub fn get_available_port() -> Option<u16> {
        (8000..(12000))
            .filter(|port| port_is_available(*port))
            .collect::<Vec<u16>>()
            .first()
            .cloned()
    }

    fn port_is_available(port: u16) -> bool {
        match std::net::TcpListener::bind(("127.0.0.1", port)) {
            Ok(_) => true,
            Err(_) => false,
        }
    }
}
