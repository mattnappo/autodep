# autodep
Distributed and automatic ML model deployment at scale

## Tasks
- [ ] Make and train a few small models in pytorch ([see this](https://pytorch.org/vision/stable/models.html))
    - [ ] Image classification
        - [x] resnet18
        - [ ] resnet50
    - [ ] semantic segmentation
        - [ ] deeplab v3
    - [ ] object detection
        - [ ] faster R-CNN
    - [ ] transformer
        - [ ] huggingface roberta exported to torchscript ([see this](https://huggingface.co/docs/transformers/torchscript))
- [x] Pytorch loading
    - [x] Lib Func to load torchscript
    - [x] Lib Func to run
- [x] Worker code
    - [x] On init:
        - [x] load pytorch model
        - [x] start RPC server listening for requests
    - [x] Creates RPC server that listens for requests to:
        - [x] Run Inference on some input (input needs to be serialized as RPC)
        - [x] return its status
        - [x] "Gracefully" be shut down
    - [x] On inference handler:
        - [x] Run inference on the model
    - [x] on shutdown handler:
        - [x] terminate itself
    - [ ] Make proper containerization system (optional)
        - [ ] Will allow extra files to be copied in to the container (do not allow this in v1)
- [x] Resource allocator / worker manager
    - [x] keep track of how many workers are idle/working
    - [x] Automatically pick an idle worker
        - [x] if all workers are busy, make a new one or enter a queue (have a MAX WORKERS value)
    - [x] Make manager spin up a new worker if it doesn't have an idle one on inference request
    - [x] start AND RUN a worker when necessary
    - [x] lib code to start a new worker
    - [x] lib code to kill workers
- [ ] Autoscaling algorithm (optional)
    - [ ] Look at past fixed window of HTTP traffic and determine how many workers to spin up/shut down
- [x] HTTP server
    - [ ] Parse config file to determine input type
    - [x] Make a single route for inference to the model
    - [x] Start the HTTP server
    - [x] HTTP server has a ResourceAllocator and will call RA methods to spin up/shut down
- [x] Main binary
    - [x] Run HTTP server
- [ ] Benchmarking suite
    - [ ] Schedule maker
    - [ ] Request schedule executor
    - [ ] Capture metrics / statistics
- [ ] Write final paper report

## Sources
* [Running TorchScript](https://github.com/LaurentMazare/tch-rs/blob/main/examples/jit/README.md)
* [Actix Twitter Clone](https://hub.qovery.com/guides/tutorial/create-a-blazingly-fast-api-in-rust-part-1/)
* [Actix Image uploads](https://www.reddit.com/r/rust/comments/xzrznn/how_to_upload_and_download_files_through_actix_web/)
* [gRPC in Rust](https://github.com/hyperium/tonic/blob/master/examples/routeguide-tutorial.md)

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
- [ ] Maybe remove second layer of indirection `Server` around `Manager`
- [ ] Make image deserialization on web side more robust

## Important improvements
- [ ] in `start_new_worker` -- a better mechanism than `thread.sleep`
- [ ] in `run_inference` -- a better mechanism for starting a new worker



## Idea
-- remove second layer of ind to prevent deadlock in multithreade testing example


## IMPORTANT
The bug seems to only happen whenever it receives two inference requests at the same time.

Seemingly can't handle a status request while inference is actually running (or, the sleep in the torch.run method is running)



its not the model inference that takes a long time.... its the new process creator (500ms?)
