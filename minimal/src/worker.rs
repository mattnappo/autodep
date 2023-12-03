use tonic::{transport::Server, Request, Response, Status};

// Import the generated rust code into module
pub mod myservice {
    tonic::include_proto!("myservice"); // The string specified here must match the proto package name
}

// Proto generated server traits
use myservice::my_grpc_service_server::{MyGrpcService, MyGrpcServiceServer};

pub struct MyGrpcServiceImpl;

#[tonic::async_trait]
impl MyGrpcService for MyGrpcServiceImpl {
    async fn infer(
        &self,
        _request: Request<myservice::InferRequest>,
    ) -> Result<Response<myservice::InferResponse>, Status> {
        Ok(Response::new(myservice::InferResponse {
            output: "this is some inference output".to_string(),
        }))
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    tracing_subscriber::fmt::init();

    let args: Vec<String> = std::env::args().collect();
    let port: u16 = args[1].parse()?;
    let addr = "[::1]:{port}".parse()?;
    let my_grpc_service = MyGrpcServiceImpl {};

    println!("MyGrpcServiceServer listening on {}", addr);

    Server::builder()
        .add_service(MyGrpcServiceServer::new(my_grpc_service))
        .serve(addr)
        .await?;

    //std::fs::File::create("./tmp/{port}_ready").unwrap();
    std::fs::write(format!("./tmp/{}_ready", port), "")?;

    Ok(())
}
