use actix_web::{web, App, HttpServer, Responder};
use rand::Rng;
use serde::{ser::Serialize, Serializer};

use std::fs;
use tokio::time::{timeout, Duration};

use std::collections::HashMap;
use std::process::Command;
use std::sync::Mutex;
use tonic::transport::Channel;
use tonic::Request;
use uuid::Uuid;

const WORKER: &str = "/home/matt/rust/autodep/minimal/target/debug/worker";

mod rpc {
    tonic::include_proto!("myservice");
}

use rpc::my_grpc_service_client::MyGrpcServiceClient;
use rpc::my_grpc_service_server::{MyGrpcService, MyGrpcServiceServer};
use rpc::{InferRequest as MyRequest, InferResponse as MyResponse};

use std::sync::Arc;

impl Serialize for MyResponse {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        serializer.collect_str(&format!("{:?}", self))
    }
}
impl Serialize for MyRequest {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        serializer.collect_str(&format!("{:?}", self))
    }
}

// Define your gRPC service client
#[derive(Clone)]
pub struct WorkerClient {
    client: Arc<Mutex<MyGrpcServiceClient<Channel>>>,
}

impl WorkerClient {
    async fn new(address: String) -> Result<Self, Box<dyn std::error::Error>> {
        let client = MyGrpcServiceClient::connect(address).await?;
        Ok(WorkerClient {
            client: Arc::new(Mutex::new(client)),
        })
    }

    async fn infer(
        &mut self,
        request: Request<MyRequest>,
    ) -> Result<MyResponse, Box<dyn std::error::Error>> {
        let mut client = self.client.lock().unwrap();
        let res = client.infer(request).await?;
        Ok(res.into_inner())
    }
}

use serde::Serialize as DeriveSerialize;
#[derive(Debug, DeriveSerialize, PartialEq)]
enum Status {
    Working = 0,
    Idle = 1,
}

// Define your worker manager
pub struct WorkerManager {
    workers: Mutex<HashMap<String, (WorkerClient, Status)>>,
}

impl WorkerManager {
    fn new() -> Self {
        WorkerManager {
            workers: Mutex::new(HashMap::new()),
        }
    }

    async fn start_worker(&self) -> Result<String, Box<dyn std::error::Error>> {
        let worker_id = Uuid::new_v4().to_string();
        let mut rng = rand::thread_rng();
        let port: u16 = rng.gen_range(9000, 11000);

        Command::new(WORKER).arg(port.to_string()).spawn()?;

        // Wait for the worker to be ready
        /*
        let wait_for_worker = async {
            loop {
                tokio::time::sleep(Duration::from_millis(100)).await;
                if std::fs::metadata(format!("./tmp/{}_ready", port)).is_ok() {
                    break;
                }
            }
        };
        timeout(Duration::from_millis(1000), wait_for_worker).await?;
        */

        /*
            let timeout_duration = Duration::from_secs(5);

            match timeout(timeout_duration, connection_attempt).await {
                Ok(stream) => {
                    // Connection succeeded
                    println!("Connected to the server: {:?}", stream);
                }
                Err(err) => {
                    // Connection attempt timed out or encountered an error
                    eprintln!("Connection error: {:?}", err);
                }
            }

        */

        // Wait for the worker to be ready
        loop {
            tokio::time::sleep(Duration::from_millis(1000)).await;
            if fs::read_to_string(format!("./tmp/{}_ready", worker_id)).is_ok() {
                break;
            }
        }

        //
        //
        //

        let worker_client = WorkerClient::new(format!("http://localhost:{}", port)).await?;
        self.workers
            .lock()
            .unwrap()
            .insert(worker_id.clone(), (worker_client, Status::Idle));
        Ok(worker_id)
    }

    fn get_idle_worker(&self) -> Option<WorkerClient> {
        // Implement your logic to find an idle worker
        let workers = self.workers.lock().unwrap();
        workers
            .values()
            .filter(|(worker, status)| *status == Status::Idle)
            .map(|(worker, _)| worker.clone())
            .collect::<Vec<WorkerClient>>()
            .first()
            .cloned()
    }
}

// Define your HTTP server
async fn handle_inference_request(worker_manager: web::Data<WorkerManager>) -> impl Responder {
    let mut worker = match worker_manager.get_idle_worker() {
        Some(worker) => worker,
        None => {
            let worker_id = worker_manager.start_worker().await.unwrap();
            let x = (*worker_manager).workers.lock().unwrap();
            let (worker, _) = x.get(&worker_id).unwrap();
            worker.clone()
        }
    };

    let response = worker
        .infer(Request::new(MyRequest {
            input: "some input".to_string(),
        }))
        .await
        .unwrap();
    web::Json(response)
}

#[actix_web::main]
async fn main() -> std::io::Result<()> {
    let worker_manager = web::Data::new(WorkerManager::new());

    HttpServer::new(move || {
        App::new()
            .app_data(worker_manager.clone())
            .route("/", web::get().to(handle_inference_request))
    })
    .bind("127.0.0.1:8080")?
    .run()
    .await
}
