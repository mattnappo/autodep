use autodep::manager;

/// This is test code right now
#[tokio::main]
async fn main() {
    // Start the HTTP server

    // Start the manager
    let manager = manager::Manager::new();
    manager.test(8000).await.unwrap();
}
