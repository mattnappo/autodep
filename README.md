# Autodep

Autodep is an easy to use tool that automates the deployment of PyTorch models. Autodep dynamically adjusts computational resources in real-time to the volume of incoming inference requests. Autodep's distributed approach scatters workload across multiple workers, providing higher request throughput.

## Features

- **Automated Deployment**: Easily deploy TorchScript models with minimal setup.
- **Distributed Architecture**: Workload is distributed across multiple workers.
- **Dynamic Scaling**: Compute resources are adjusted in real-time based on inference request volume.
- **Memory Safety and Performance**: Built in Rust, ensuring safety, low resource consumption, and high performance.
- **User-Friendly**: Simple interface requiring just a TorchScript file to start an API server for inference.

## Usage

### Configuring

The default configuration is accessible in `config.toml`. Note that Autodep requires a local installation of `libtorch`, so please install `libtorch` before running autodep. 

### Compiling

After `libtorch` has been installed and the path is set in the config file, compile Autodep with
```
cargo build --release
```

### Running

To use Autodep, provide a TorchScript file, and the tool will start an HTTP server that listens for JSON-encoded POST requests at the `/inference` endpoint.

Usage:

```
cargo run --release --bin autodep <config file> <model file>
```
or
```
./autodep <config file> <model file>
```

This command starts an HTTP server on the port specified in the config file. This server serves the TorchScript model located at `<model file>` via the `/inference` route.

Note: make sure that there is a `logs/` folder in the current directory.

## Routes

### POST `/inference`
Run model inference

For TextToText or ImageToImage inference, requests are of type:
```json
{
    "data": {
        "B64Image": {
            "image": "<base-64 encoded image>",
            "height": 224,
            "width": 224
        }
    },
    "inference_type": "ImageToImage"
}
```
or
```json
{
    "data": {
        "text": "<input text>"
    },
    "inference_type": "TextToText"
}
```

For ImageClassification tasks, requests are of type:
```json
{
    "data": {
        "B64Image": {
            "image": "<base-64 encoded image>",
            "height": 500,
            "width": 700
        }
    },
    "inference_type": {
        "ImageClassification": {
            "top_n": 5
        }
    }
}
```

where `top_n` is a parameter for the number of top classes to return.

Example requests can be found in `/tests/`.

### GET `/workers`
View the currently-active workers

## Auxiliary Routes

### GET `/workers/_info`
View statistics such as number of requests served for each worker

### GET `/workers/_status`
View the status of the workers

## Documentation

Documentation is available at
```
https://mattnappo.github.io/docs/autodep
```

You can view the documentation locally after running `cargo doc`. Then, visit `target/doc/autodep/index.html`.

## System Architecture

Autodep's architecture is composed of an HTTP server, a Worker Manager, and a cluster of Worker nodes. The HTTP server acts as the user interface, while the Worker Manager is responsible for managing the workers, routing requests, and handling resource allocation. Workers themselves run an RPC server providing services for executing model inference.

## License

Autodep is open-source software licensed under the GNU General Public License v3.0.

