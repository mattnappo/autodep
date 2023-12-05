#1/bin/bash
RUST_BACKTRACE=1 cargo build --release && cargo run --release --bin autodep config.toml $1
