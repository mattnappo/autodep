#!/bin/bash

cargo build

RUST_LOG=debug cargo run --bin autodep ./models/resnet18.pt
