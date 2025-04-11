.PHONY: all build clean run test rust go examples

# Default target
all: build

# Build both Rust and Go components
build: rust go

# Build just the Rust library
rust:
	@echo "Building Rust library..."
	cd rust && cargo build --release

# Build just the Go application
go:
	@echo "Building Go application..."
	go build -o bert_classifier ./cmd/bert

# Build examples
examples: rust
	@echo "Building examples..."
	go build -o examples/simple_classification ./examples/simple_classification.go

# Run the application with the correct library path
run: build
	@echo "Running BERT classifier..."
	LD_LIBRARY_PATH=./rust/target/release ./bert_classifier

# Run the simple classification example
run-example: examples
	@echo "Running simple classification example..."
	LD_LIBRARY_PATH=./rust/target/release ./examples/simple_classification

# Clean built artifacts
clean:
	@echo "Cleaning build artifacts..."
	rm -f bert_classifier
	rm -f examples/simple_classification
	cd rust && cargo clean

# Run tests
test:
	@echo "Running tests..."
	cd rust && cargo test
	go test ./...

# Install the application to $GOPATH/bin
install: build
	@echo "Installing bert_classifier to GOPATH/bin..."
	cp bert_classifier $(GOPATH)/bin/

# Generate documentation
docs:
	@echo "Generating documentation..."
	cd rust && cargo doc --no-deps
	go doc -all ./...

# Show help information
help:
	@echo "Candle BERT Go Makefile"
	@echo "----------------------"
	@echo "make              - Build both Rust and Go components"
	@echo "make rust         - Build only the Rust library"
	@echo "make go           - Build only the Go application"
	@echo "make examples     - Build example applications"
	@echo "make run          - Build and run the main application"
	@echo "make run-example  - Build and run the simple example"
	@echo "make clean        - Remove build artifacts"
	@echo "make test         - Run tests"
	@echo "make install      - Install to GOPATH/bin"
	@echo "make docs         - Generate documentation" 