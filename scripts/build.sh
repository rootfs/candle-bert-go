#!/bin/bash
set -e

# Build the Rust library
echo "Building Rust library..."
cd "$(dirname "$0")/../rust"
cargo build --release

# Build the Go application
echo "Building Go application..."
cd ..
go build -o bert_classifier ./cmd/bert

echo "Done! You can run the application with:"
echo "LD_LIBRARY_PATH=./rust/target/release ./bert_classifier" 