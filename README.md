# autodep
Distributed and automatic ML model deployment at scale

## Todo
- [ ] Make and train a few small models in pytorch
    - [x] resnet
    - [ ] transformer
- [x] Pytorch loading
    - [x] Lib Func to load torchscript
    - [x] Lib Func to run
- [ ] Worker code
    - [x] On init:
        - [x] load pytorch model
        - [x] start RPC server listening for requests
    - [x] Creates RPC server that listens for requests to:
        - [ ] Run Inference on some input (input needs to be serialized as RPC)
        - [x] return its status
        - [x] "Gracefully" be shut down
    - [ ] On inference handler:
        - [ ] Run inference on the model
    - [x] on shutdown handler:
        - [x] terminate itself
    - [ ] Make proper containerization system (optional)
        - [ ] Will allow extra files to be copied in to the container (do not allow this in v1)
- [ ] Resource allocator / worker manager
    - [x] keep track of how many workers are idle/working
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
- [ ] All `todo!()` and `TODO` code
- [ ] Support more input datatypes
- [ ] Make `TorchModel` more robust (make input type declared when `new` is called)
- [ ] Explicit CPU/GPU support
- [ ] Don't hardcode for resnet
- [ ] Make `Class` require one or the other option
- [ ] Make `TOP_N` an HTTP API parameter
- [ ] Add stats to `Worker`
- [ ] AWS (S3) input/output support
- [ ] Make `TResult` cleaner type alias
- [ ] Allow for `Worker`s on different hosts
- [ ] Make a better mechanism than `thread.sleep` for waiting for a worker to start
    - [ ] Idea: make the worker return `Ready` when its done initializing?
    - [ ] Idea: have a timeout mechanism?
- [ ] Run new `./worker` processes procs without cargo
- [ ] Fix the jank in `all_status`
- [ ] Make `class_int` optional in the protobuf
