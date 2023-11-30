# autodep
Distributed and automatic ML model deployment at scale

## Requirements
- `libtorch` version 2.1.0

## Todo
- [ ] Make and train a few small models in pytorch
- [ ] Pytorch loading
    - [ ] Lib Func to load torchscript
    - [ ] Lib Func to run
- [ ] Worker code
    - [ ] On init:
        - [ ] load pytorch model
        - [ ] start RPC server listening for requests
    - [ ] Creates RPC server that listens for requests to:
        - [ ] Run Inference on some input (input needs to be serialized as RPC)
        - [ ] "Gracefully" be shut down
    - [ ] On inference handler:
        - [ ] Run inference on the model
    - [ ] on shutdown handler:
        - [ ] terminate itself
    - [ ] Make proper containerization system (optional)
        - [ ] Will allow extra files to be copied in to the container (do not allow this in v1)
- [ ] Resource allocator / worker manager
    - [ ] keep track of how many workers are idle/working
    - [ ] Automatically pick an idle worker
        - [ ] if all workers are busy, make a new one or enter a queue (have a MAX WORKERS value)
    - [ ] start AND RUN a worker when necessary
    - [ ] start idle workers
    - [ ] kill workers
- [ ] Autoscaling algorithm (optional)
    - [ ] Look at past fixed window of HTTP traffic and determine how many workers to spin up/shut down
- [ ] HTTP server
    - [ ] Parse config file to determine input type
    - [ ] Make a single route for inference to the model
    - [ ] Start the HTTP server
    - [ ] HTTP server has a ResourceAllocator and will call RA methods to spin up/shut down
- [ ] Main binary
    - [ ] Run HTTP server
- [ ] Benchmarking suite
    - [ ] Schedule maker
    - [ ] Request schedule executor
    - [ ] Capture metrics / statistics
- [ ] Write final paper

## Loading and running torch
[Source](https://github.com/LaurentMazare/tch-rs/blob/main/examples/jit/README.md)

## Future Improvements
- [ ] Support more input datatypes
- [ ] Make `TorchLoader` more robust (make input type declared when `new` is  called)
- [ ] Explicit CPU/GPU support
