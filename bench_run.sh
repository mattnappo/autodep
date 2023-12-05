#1/bin/bash
RUST_BACKTRACE=1 cargo build --release && cargo run --release --bin autodep bench_config.toml $1
