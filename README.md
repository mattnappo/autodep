# Autodep

Autodep is an easy to use tool that automates the deployment of PyTorch models. Autodep dynamically adjusts computational resources in real-time to the volume of incoming inference requests. Autodep's distributed approach scatters workload across multiple workers, providing higher request throughput.

## Features

- **Automated Deployment**: Easily deploy TorchScript models with minimal setup.
- **Distributed Architecture**: Workload is distributed across multiple workers.
- **Dynamic Scaling**: Compute resources are adjusted in real-time based on inference request volume.
- **Memory Safety and Performance**: Built in Rust, ensuring safety, low resource consumption, and high performance.
- **User-Friendly**: Simple interface requiring just a TorchScript file to start an API server for inference.

## Usage

### Compiling
```
cargo build --release
```

### Configuring



### Running

To use Autodep, provide a TorchScript file, and the tool will start an HTTP server that listens for JSON-encoded POST requests at the `/inference` endpoint.

Usage:

```
./autodep <port> <model file>
```

This command starts an HTTP server on port `<port>` that serves the TorchScript model located at `<model file>`.

## Documentation

You can view the documentation locally after running `cargo doc`. Then, visit `target/doc/autodep/index.html`.

## System Architecture

Autodep's architecture is composed of an HTTP server, a Worker Manager, and a cluster of Worker nodes. The HTTP server acts as the user interface, while the Worker Manager is responsible for managing the workers, routing requests, and handling resource allocation. Workers themselves run an RPC server providing services for executing model inference.

## License

Autodep is open-source software licensed under the GNU General Public License v3.0.

